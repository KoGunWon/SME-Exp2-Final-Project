import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn


# =======================================================
# 1. AI 모델 구조: train.py와 동일하게 유지
# =======================================================
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


# =======================================================
# 2. 유틸리티 함수
# =======================================================
def _resolve_path(candidates):
    """현재 작업 폴더와 main.py가 있는 폴더를 모두 탐색한다."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for name in candidates:
        cwd_path = os.path.join(os.getcwd(), name)
        if os.path.exists(cwd_path):
            return cwd_path

        script_path = os.path.join(base_dir, name)
        if os.path.exists(script_path):
            return script_path

    raise FileNotFoundError(f"다음 파일 중 하나를 찾을 수 없습니다: {candidates}")


def _normalize_bs_positions(p_bs):
    """앵커 좌표를 항상 (num_anchors, 2) 형태로 변환한다."""
    p_bs = np.asarray(p_bs, dtype=float)

    if p_bs.ndim != 2:
        raise ValueError(f"p_bs 또는 BS_positions는 2차원 배열이어야 합니다. 현재 shape: {p_bs.shape}")

    if p_bs.shape[1] == 2:
        return p_bs
    if p_bs.shape[0] == 2:
        return p_bs.T

    raise ValueError(f"앵커 좌표 shape를 해석할 수 없습니다. 현재 shape: {p_bs.shape}")


def _normalize_d_hat(d_hat, num_anchors):
    """거리 행렬을 항상 (num_anchors, num_user) 형태로 변환한다."""
    d_hat = np.asarray(d_hat, dtype=float)

    if d_hat.ndim != 2:
        raise ValueError(f"d_hat은 2차원 배열이어야 합니다. 현재 shape: {d_hat.shape}")

    if d_hat.shape[0] == num_anchors:
        return d_hat
    if d_hat.shape[1] == num_anchors:
        return d_hat.T

    raise ValueError(
        f"d_hat shape가 앵커 수와 맞지 않습니다. d_hat shape: {d_hat.shape}, 앵커 수: {num_anchors}"
    )


# =======================================================
# 3. GERT 전처리 함수: train.py와 동일한 핵심 로직 적용
# =======================================================
def apply_mad_clipping(d_hat, num_anchors, threshold=3.5):
    """train.py의 MAD 기반 이상치 clipping과 동일한 방식."""
    median_d = np.median(d_hat, axis=1, keepdims=True)
    mad = np.median(np.abs(d_hat - median_d), axis=1, keepdims=True)
    mad = np.where(mad == 0, 1e-6, mad)

    mod_z_score = 0.6745 * (d_hat - median_d) / mad
    clipped_d_hat = np.copy(d_hat)

    for i in range(num_anchors):
        outlier_mask = np.abs(mod_z_score[i]) > threshold
        if np.any(outlier_mask):
            valid_data = d_hat[i, ~outlier_mask]
            if len(valid_data) > 0:
                clipped_d_hat[i, outlier_mask] = np.clip(
                    d_hat[i, outlier_mask],
                    np.min(valid_data),
                    np.max(valid_data),
                )

    return clipped_d_hat


def get_gert_features(d_hat_raw, p_bs):
    """
    GERT 입력 feature 생성.
    - 입력 거리 d_hat_raw: (num_anchors, num_user) 또는 (num_user, num_anchors)
    - 입력 앵커 p_bs: (2, num_anchors) 또는 (num_anchors, 2)
    - 출력 X_tensor: (num_user, num_anchors, 2)
    - 출력 initial_guesses: (num_user, 2)
    """
    bs_pos = _normalize_bs_positions(p_bs)
    num_anchors = bs_pos.shape[0]
    d_hat = _normalize_d_hat(d_hat_raw, num_anchors)

    # 1. 학습 코드와 동일한 MAD clipping 적용
    d_hat_clipped = apply_mad_clipping(d_hat, num_anchors=num_anchors, threshold=3.5)

    # 2. 초기 위치 추정: inverse-distance weighted centroid
    inv_d = 1.0 / (d_hat_clipped + 1e-6)
    weights = inv_d / np.sum(inv_d, axis=0, keepdims=True)
    initial_guesses = weights.T @ bs_pos

    # 3. 기하학적 residual 생성
    num_user = d_hat_clipped.shape[1]
    residual_features = np.zeros((num_user, num_anchors))

    for i in range(num_user):
        guess_pos = initial_guesses[i]
        physical_dist = np.linalg.norm(bs_pos - guess_pos, axis=1)
        residual_features[i, :] = d_hat_clipped[:, i] - physical_dist

    # 4. Transformer 입력: 각 앵커를 하나의 token으로 보고 [거리, residual]을 feature로 구성
    X_seq = np.stack([d_hat_clipped.T, residual_features], axis=-1)
    return torch.tensor(X_seq, dtype=torch.float32), initial_guesses


def _load_state_dict_compatible(model_path):
    """PyTorch 버전에 따른 weights_only 인자 호환 처리."""
    try:
        return torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location=torch.device("cpu"))


# =======================================================
# 4. 메인 함수
# =======================================================
def main():
    # 1. 데이터 로드
    mat_path = _resolve_path(["DH_FR1.mat", "InF_DH_FR1.mat"])
    data = sio.loadmat(mat_path, squeeze_me=False)

    if "p_bs" in data:
        p_bs = data["p_bs"]
    elif "BS_positions" in data:
        p_bs = data["BS_positions"]
    else:
        raise KeyError("MAT 파일에 'p_bs' 또는 'BS_positions' 변수가 없습니다.")

    if "d_hat" not in data:
        raise KeyError("MAT 파일에 'd_hat' 변수가 없습니다.")

    d_hat = data["d_hat"]

    # 2. train.py와 동일한 방식으로 feature 생성
    X_tensor, initial_guesses = get_gert_features(d_hat, p_bs)

    # 3. 모델 로드 및 AI residual 보정
    model = CompactResidualTransformer(num_anchors=X_tensor.shape[1])
    model_path = _resolve_path(["model.pt"])

    try:
        state_dict = _load_state_dict_compatible(model_path)
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            predicted_residuals = model(X_tensor).cpu().numpy()

        # 최종 위치 = 초기 추정 위치 + AI 보정량
        p_hat_processed = initial_guesses + predicted_residuals

    except Exception as e:
        # 모델 로드 또는 추론 실패 시 최소한 baseline 결과라도 반환
        print(f"Warning: 모델 추론 실패. 초기 가중 중심 추정값으로 대체합니다. 원인: {e}")
        p_hat_processed = initial_guesses

    # 제출 규격: (2, num_user)
    p_hat = p_hat_processed.T
    return p_hat


if __name__ == "__main__":
    estimated_p = main()
    print(f"측위 완료. 결과 shape: {estimated_p.shape}")
