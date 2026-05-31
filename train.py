import argparse
import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# =======================================================
# 1. 설정 및 재현성
# =======================================================
@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 200
    batch_size: int = 32
    lr: float = 0.005
    weight_decay: float = 1e-4
    val_ratio: float = 0.15
    mad_threshold: float = 3.5
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 1
    dropout_encoder: float = 0.2
    dropout_head: float = 0.1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =======================================================
# 2. 파일 및 배열 shape 처리
# =======================================================
def resolve_path(candidates):
    """현재 작업 폴더와 train.py가 있는 폴더에서 후보 파일을 탐색한다."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for name in candidates:
        candidate_paths = [
            os.path.join(os.getcwd(), name),
            os.path.join(base_dir, name),
            name,
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f"다음 파일 중 하나를 찾을 수 없습니다: {candidates}")


def normalize_bs_positions(p_bs):
    """앵커 좌표를 항상 (num_anchors, 2) 형태로 변환한다."""
    p_bs = np.asarray(p_bs, dtype=np.float32)
    if p_bs.ndim != 2:
        raise ValueError(f"BS_positions 또는 p_bs는 2차원이어야 합니다. 현재 shape: {p_bs.shape}")
    if p_bs.shape[1] == 2:
        return p_bs
    if p_bs.shape[0] == 2:
        return p_bs.T
    raise ValueError(f"앵커 좌표 shape를 해석할 수 없습니다. 현재 shape: {p_bs.shape}")


def normalize_d_hat(d_hat, num_anchors):
    """거리 행렬을 항상 (num_anchors, num_user) 형태로 변환한다."""
    d_hat = np.asarray(d_hat, dtype=np.float32)
    if d_hat.ndim != 2:
        raise ValueError(f"d_hat은 2차원이어야 합니다. 현재 shape: {d_hat.shape}")
    if d_hat.shape[0] == num_anchors:
        return d_hat
    if d_hat.shape[1] == num_anchors:
        return d_hat.T
    raise ValueError(f"d_hat shape가 앵커 수와 맞지 않습니다. d_hat: {d_hat.shape}, anchors: {num_anchors}")


def normalize_true_positions(p_true, num_user):
    """정답 좌표를 항상 (num_user, 2) 형태로 변환한다."""
    p_true = np.asarray(p_true, dtype=np.float32)
    if p_true.ndim != 2:
        raise ValueError(f"p는 2차원이어야 합니다. 현재 shape: {p_true.shape}")
    if p_true.shape == (num_user, 2):
        return p_true
    if p_true.shape == (2, num_user):
        return p_true.T
    raise ValueError(f"p shape를 해석할 수 없습니다. p: {p_true.shape}, num_user: {num_user}")


def load_localization_data(data_path=None):
    if data_path is None:
        data_path = resolve_path(["DH_FR1.mat", "InF_DH_FR1.mat"])
    else:
        data_path = resolve_path([data_path])

    mat_data = sio.loadmat(data_path, squeeze_me=False)

    if "BS_positions" in mat_data:
        bs_raw = mat_data["BS_positions"]
    elif "p_bs" in mat_data:
        bs_raw = mat_data["p_bs"]
    else:
        raise KeyError("MAT 파일에 'BS_positions' 또는 'p_bs' 변수가 없습니다.")

    if "d_hat" not in mat_data:
        raise KeyError("MAT 파일에 'd_hat' 변수가 없습니다.")
    if "p" not in mat_data:
        raise KeyError("학습에는 ground truth 위치 'p'가 필요합니다.")

    bs_positions = normalize_bs_positions(bs_raw)
    d_hat = normalize_d_hat(mat_data["d_hat"], num_anchors=bs_positions.shape[0])
    p_true = normalize_true_positions(mat_data["p"], num_user=d_hat.shape[1])

    return data_path, bs_positions, d_hat, p_true


# =======================================================
# 3. GERT 전처리: main.py와 동일한 핵심 로직
# =======================================================
class GERTPreprocessor:
    def __init__(self, bs_positions, mad_threshold=3.5):
        self.bs_positions = normalize_bs_positions(bs_positions)
        self.num_anchors = self.bs_positions.shape[0]
        self.mad_threshold = mad_threshold

    def apply_mad_clipping(self, d_hat):
        """앵커별 MAD 기반 clipping. main.py와 동일한 입력 분포를 만들기 위한 전처리."""
        d_hat = normalize_d_hat(d_hat, self.num_anchors)

        median_d = np.median(d_hat, axis=1, keepdims=True)
        mad = np.median(np.abs(d_hat - median_d), axis=1, keepdims=True)
        mad = np.where(mad == 0, 1e-6, mad)

        mod_z_score = 0.6745 * (d_hat - median_d) / mad
        clipped_d_hat = np.copy(d_hat)

        for i in range(self.num_anchors):
            outlier_mask = np.abs(mod_z_score[i]) > self.mad_threshold
            if np.any(outlier_mask):
                valid_data = d_hat[i, ~outlier_mask]
                if len(valid_data) > 0:
                    clipped_d_hat[i, outlier_mask] = np.clip(
                        d_hat[i, outlier_mask],
                        np.min(valid_data),
                        np.max(valid_data),
                    )
        return clipped_d_hat.astype(np.float32)

    def get_initial_guess(self, d_hat_clipped):
        """Inverse-distance weighted centroid 초기 위치 추정."""
        inv_d = 1.0 / (d_hat_clipped + 1e-6)
        weights = inv_d / np.sum(inv_d, axis=0, keepdims=True)
        initial_guesses = weights.T @ self.bs_positions
        return initial_guesses.astype(np.float32)

    def process(self, d_hat_raw):
        """
        출력:
        - X_seq: (num_user, num_anchors, 2), 각 앵커 token의 [거리, residual]
        - initial_guesses: (num_user, 2)
        """
        d_hat_clipped = self.apply_mad_clipping(d_hat_raw)
        initial_guesses = self.get_initial_guess(d_hat_clipped)

        num_user = d_hat_clipped.shape[1]
        residual_features = np.zeros((num_user, self.num_anchors), dtype=np.float32)

        for i in range(num_user):
            guess_pos = initial_guesses[i]
            physical_dist = np.linalg.norm(self.bs_positions - guess_pos, axis=1)
            residual_features[i, :] = d_hat_clipped[:, i] - physical_dist

        X_seq = np.stack([d_hat_clipped.T, residual_features], axis=-1).astype(np.float32)
        return X_seq, initial_guesses


# =======================================================
# 4. 데이터셋 및 모델
# =======================================================
class LocalizationDataset(Dataset):
    def __init__(self, X_seq, y_delta):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y_delta, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CompactResidualTransformer(nn.Module):
    def __init__(self, num_anchors=18, feature_dim=2, d_model=32, nhead=4, num_layers=1):
        super(CompactResidualTransformer, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_anchors, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        # x: (batch, num_anchors, 2)
        x = self.embedding(x)
        x = x + self.pos_embedding
        attn_out = self.transformer(x)
        pooled_out = torch.mean(attn_out, dim=1)
        return self.fc_out(pooled_out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =======================================================
# 5. 평가 함수
# =======================================================
def compute_error_metrics(pred_pos, true_pos):
    errors = np.linalg.norm(pred_pos - true_pos, axis=1)
    return {
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "p90_error": float(np.percentile(errors, 90)),
        "max_error": float(np.max(errors)),
    }


def print_metrics(title, metrics):
    print(f"\n[{title}]")
    print(f"Mean   : {metrics['mean_error']:.4f} m")
    print(f"Median : {metrics['median_error']:.4f} m")
    print(f"RMSE   : {metrics['rmse']:.4f} m")
    print(f"P90    : {metrics['p90_error']:.4f} m")
    print(f"Max    : {metrics['max_error']:.4f} m")


def predict_delta(model, X_seq, batch_size=128, device="cpu"):
    model.eval()
    preds = []
    tensor = torch.tensor(X_seq, dtype=torch.float32)
    loader = DataLoader(tensor, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device)
            preds.append(model(X_batch).cpu().numpy())
    return np.vstack(preds)


# =======================================================
# 6. 학습 루프
# =======================================================
def train_one_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    config=None,
    save_path=None,
    distance_only=False,
    device="cpu",
):
    if config is None:
        config = TrainConfig()

    X_train_used = np.copy(X_train)
    X_val_used = None if X_val is None else np.copy(X_val)

    if distance_only:
        # ablation: residual channel을 0으로 제거하여 거리만 사용
        X_train_used[:, :, 1] = 0.0
        if X_val_used is not None:
            X_val_used[:, :, 1] = 0.0

    train_dataset = LocalizationDataset(X_train_used, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    val_loader = None
    if X_val_used is not None and y_val is not None:
        val_dataset = LocalizationDataset(X_val_used, y_val)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = CompactResidualTransformer(
        num_anchors=X_train.shape[1],
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_score = float("inf")
    best_state = None
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * X_batch.size(0)

        train_loss = train_loss_sum / len(train_dataset)
        val_loss = None

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss_sum += loss.item() * X_batch.size(0)
            val_loss = val_loss_sum / len(val_loader.dataset)
            score = val_loss
        else:
            score = train_loss

        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_path is not None:
                torch.save(best_state, save_path)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch == 1 or epoch % 40 == 0 or epoch == config.epochs:
            if val_loss is None:
                print(f"Epoch [{epoch:3d}/{config.epochs}] | Train MSE: {train_loss:.4f}")
            else:
                print(f"Epoch [{epoch:3d}/{config.epochs}] | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_score


def split_train_val(num_samples, val_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    val_size = int(round(num_samples * val_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return train_idx, val_idx


def save_history_csv(history, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss\n")
        for row in history:
            val = "" if row["val_loss"] is None else f"{row['val_loss']:.10f}"
            f.write(f"{row['epoch']},{row['train_loss']:.10f},{val}\n")


# =======================================================
# 7. 메인 실행
# =======================================================
def train_model(args=None):
    parser = argparse.ArgumentParser(description="GERT residual Transformer training script")
    parser.add_argument("--data", type=str, default=None, help="MAT 데이터 파일 경로. 기본값: DH_FR1.mat 또는 InF_DH_FR1.mat 자동 탐색")
    parser.add_argument("--output-dir", type=str, default=".", help="model.pt와 metric 파일 저장 폴더")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mad-threshold", type=float, default=3.5)
    parser.add_argument("--no-final", action="store_true", help="검증용 학습만 수행하고 전체 데이터 재학습은 생략")
    parser.add_argument("--ablation", action="store_true", help="residual channel 제거 ablation도 추가 수행")
    parser.add_argument("--num-threads", type=int, default=1, help="CPU 학습 시 사용할 PyTorch thread 수")
    parsed = parser.parse_args(args)

    config = TrainConfig(
        seed=parsed.seed,
        epochs=parsed.epochs,
        batch_size=parsed.batch_size,
        lr=parsed.lr,
        weight_decay=parsed.weight_decay,
        val_ratio=parsed.val_ratio,
        mad_threshold=parsed.mad_threshold,
    )

    if parsed.num_threads is not None and parsed.num_threads > 0:
        torch.set_num_threads(parsed.num_threads)
        try:
            torch.set_num_interop_threads(parsed.num_threads)
        except RuntimeError:
            pass

    set_seed(config.seed)
    os.makedirs(parsed.output_dir, exist_ok=True)

    data_path, bs_positions, d_hat, p_true = load_localization_data(parsed.data)
    num_anchors, num_user = d_hat.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("GERT Training")
    print("=" * 60)
    print(f"Data path       : {data_path}")
    print(f"BS_positions   : {bs_positions.shape}")
    print(f"d_hat           : {d_hat.shape}")
    print(f"p_true          : {p_true.shape}")
    print(f"Device          : {device}")
    print(f"Seed            : {config.seed}")
    print(f"Epochs          : {config.epochs}")
    print(f"Torch threads   : {torch.get_num_threads()}")
    print(f"Val ratio       : {config.val_ratio}")

    tmp_model = CompactResidualTransformer(num_anchors=num_anchors)
    print(f"Trainable params: {count_parameters(tmp_model):,}")

    train_idx, val_idx = split_train_val(num_user, config.val_ratio, config.seed)
    print(f"Train samples   : {len(train_idx)}")
    print(f"Val samples     : {len(val_idx)}")

    # 검증용 전처리: split 이후 수행하여 p 라벨 누수 없이 검증 지표를 계산한다.
    preprocessor = GERTPreprocessor(bs_positions, mad_threshold=config.mad_threshold)

    X_train, init_train = preprocessor.process(d_hat[:, train_idx])
    y_train = p_true[train_idx] - init_train

    X_val, init_val = preprocessor.process(d_hat[:, val_idx])
    y_val = p_true[val_idx] - init_val

    baseline_val_metrics = compute_error_metrics(init_val, p_true[val_idx])
    print_metrics("Validation baseline: inverse-distance weighted centroid", baseline_val_metrics)

    print("\n--- Validation training: GERT full feature [distance, residual] ---")
    val_model_path = os.path.join(parsed.output_dir, "model.pt")
    model, history, best_val_loss = train_one_model(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
        save_path=val_model_path,
        distance_only=False,
        device=device,
    )

    pred_delta_val = predict_delta(model, X_val, batch_size=128, device=device)
    pred_pos_val = init_val + pred_delta_val
    gert_val_metrics = compute_error_metrics(pred_pos_val, p_true[val_idx])
    print_metrics("Validation GERT full", gert_val_metrics)

    metrics = {
        "config": asdict(config),
        "data_path": data_path,
        "num_anchors": int(num_anchors),
        "num_user": int(num_user),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "trainable_parameters": int(count_parameters(tmp_model)),
        "best_val_mse": float(best_val_loss),
        "validation_baseline": baseline_val_metrics,
        "validation_gert_full": gert_val_metrics,
    }

    save_history_csv(history, os.path.join(parsed.output_dir, "history_val_full.csv"))

    if parsed.ablation:
        print("\n--- Validation ablation: Transformer distance-only, residual channel removed ---")
        ablation_model_path = os.path.join(parsed.output_dir, "model_val_distance_only.pt")
        model_ab, history_ab, best_ab_loss = train_one_model(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
            save_path=ablation_model_path,
            distance_only=True,
            device=device,
        )
        X_val_distance_only = np.copy(X_val)
        X_val_distance_only[:, :, 1] = 0.0
        pred_delta_ab = predict_delta(model_ab, X_val_distance_only, batch_size=128, device=device)
        pred_pos_ab = init_val + pred_delta_ab
        ablation_metrics = compute_error_metrics(pred_pos_ab, p_true[val_idx])
        print_metrics("Validation distance-only ablation", ablation_metrics)
        metrics["best_val_mse_distance_only"] = float(best_ab_loss)
        metrics["validation_distance_only_ablation"] = ablation_metrics
        save_history_csv(history_ab, os.path.join(parsed.output_dir, "history_val_distance_only.csv"))

    if not parsed.no_final:
        print("\n--- Final training: all 700 samples, save model.pt for main.py ---")
        # 최종 제출용 모델은 전체 학습 데이터를 사용한다. hidden test용 main.py도 같은 전처리 흐름을 사용한다.
        X_all, init_all = preprocessor.process(d_hat)
        y_all = p_true - init_all

        final_model_path = os.path.join(parsed.output_dir, "model.pt")
        final_model, history_final, best_train_loss = train_one_model(
            X_all,
            y_all,
            X_val=None,
            y_val=None,
            config=config,
            save_path=final_model_path,
            distance_only=False,
            device=device,
        )

        pred_delta_all = predict_delta(final_model, X_all, batch_size=128, device=device)
        pred_pos_all = init_all + pred_delta_all
        baseline_all_metrics = compute_error_metrics(init_all, p_true)
        final_all_metrics = compute_error_metrics(pred_pos_all, p_true)

        print_metrics("All-data baseline: inverse-distance weighted centroid", baseline_all_metrics)
        print_metrics("All-data GERT final model", final_all_metrics)

        metrics["best_train_mse_final"] = float(best_train_loss)
        metrics["all_data_baseline"] = baseline_all_metrics
        metrics["all_data_gert_final"] = final_all_metrics
        save_history_csv(history_final, os.path.join(parsed.output_dir, "history_final_full.csv"))
        print(f"\nSaved final model: {final_model_path}")
    else:
        print(f"\nSaved validation best model: {val_model_path}")

    metrics_path = os.path.join(parsed.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics: {metrics_path}")

    return metrics


if __name__ == "__main__":
    train_model()
