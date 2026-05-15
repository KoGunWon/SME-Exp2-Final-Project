import os
import random
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =======================================================
# 1. 시드 고정 (채점 시 완벽한 재현성 확보)
# =======================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =======================================================
# 2. 전처리 클래스 (이상치 클리핑 및 기하학적 잔차 추출)
# =======================================================
class GERTPreprocessor:
    def __init__(self, bs_positions):
        self.bs_positions = bs_positions.T if bs_positions.shape[0] == 2 else bs_positions
        self.num_anchors = self.bs_positions.shape[0]
        
    def _apply_mad_clipping(self, d_hat, threshold=3.5):
        median_d = np.median(d_hat, axis=1, keepdims=True)
        mad = np.median(np.abs(d_hat - median_d), axis=1, keepdims=True)
        mad = np.where(mad == 0, 1e-6, mad)
        mod_z_score = 0.6745 * (d_hat - median_d) / mad
        clipped_d_hat = np.copy(d_hat)
        
        for i in range(self.num_anchors):
            outlier_mask = np.abs(mod_z_score[i]) > threshold
            if np.any(outlier_mask):
                valid_data = d_hat[i, ~outlier_mask]
                if len(valid_data) > 0:
                    clipped_d_hat[i, outlier_mask] = np.clip(d_hat[i, outlier_mask], np.min(valid_data), np.max(valid_data))
        return clipped_d_hat

    def _get_initial_guess(self, d_hat):
        inv_d = 1.0 / (d_hat + 1e-6) 
        weights = inv_d / np.sum(inv_d, axis=0) 
        return (weights.T @ self.bs_positions) 

    def process(self, d_hat_raw):
        d_hat_clipped = self._apply_mad_clipping(d_hat_raw)
        initial_guesses = self._get_initial_guess(d_hat_clipped)
        
        N = d_hat_clipped.shape[1]
        residual_features = np.zeros((N, self.num_anchors))
        
        for i in range(N):
            guess_pos = initial_guesses[i]
            physical_dist = np.linalg.norm(self.bs_positions - guess_pos, axis=1)
            residual_features[i, :] = d_hat_clipped[:, i] - physical_dist
            
        X = np.hstack([d_hat_clipped.T, residual_features])
        return X, initial_guesses

# =======================================================
# 3. 데이터셋 클래스 (Transformer 시퀀스 입력용 변환)
# =======================================================
class LocalizationDataset(Dataset):
    def __init__(self, X, y):
        self.N = X.shape[0]
        # [중요 참신성] 36개의 특징을 (18개 앵커, 2개 특징)의 시퀀스로 재배열
        # 각 앵커를 NLP의 '단어(Token)'처럼 취급하기 위함
        distances = X[:, :18]
        residuals = X[:, 18:]
        X_seq = np.stack([distances, residuals], axis=-1) # (N, 18, 2)
        
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =======================================================
# 4. AI 모델: Compact Residual Transformer
# =======================================================
class CompactResidualTransformer(nn.Module):
    def __init__(self, num_anchors=18, feature_dim=2, d_model=32, nhead=4, num_layers=1):
        super(CompactResidualTransformer, self).__init__()
        # 앵커별(거리, 오차) 특징을 32차원으로 임베딩
        self.embedding = nn.Linear(feature_dim, d_model)
        
        # 앵커 번호(위치)를 기억하기 위한 Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_anchors, d_model))
        
        # [과적합 방지] 아주 얕은 Transformer Encoder (Layer=1) + Dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=64, 
            dropout=0.2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 최종 오차(Delta X, Delta Y) 예측
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2)
        )
        
    def forward(self, x):
        # x shape: (Batch, 18, 2)
        x = self.embedding(x) # (Batch, 18, 32)
        x = x + self.pos_embedding 
        
        # Self-Attention 수행: AI가 신뢰할 수 있는 앵커를 스스로 파악
        attn_out = self.transformer(x) # (Batch, 18, 32)
        
        # 18개 앵커의 정보를 하나로 압축 (Global Average Pooling)
        pooled_out = torch.mean(attn_out, dim=1) # (Batch, 32)
        
        # 최종 보정값(Delta) 예측
        out = self.fc_out(pooled_out) # (Batch, 2)
        return out

# =======================================================
# 5. 메인 학습 루프 (Training)
# =======================================================
def train_model():
    set_seed(42) # 재현성 보장
    
    # 데이터 로드 
    data_path = 'DH_FR1.mat'
        
    print(f"데이터 로딩 중... ({data_path})")
    mat_data = scipy.io.loadmat(data_path)
    bs_positions = mat_data['BS_positions']
    d_hat = mat_data['d_hat']
    p_true = mat_data['p']
    
    # 전처리 및 타겟 설정
    preprocessor = GERTPreprocessor(bs_positions)
    X_features, initial_guesses = preprocessor.process(d_hat)
    y_target = p_true.T - initial_guesses # 모델이 학습할 '잔차 오차'
    
    # [과적합 방지] Train / Val Split (85% 학습, 15% 검증)
    dataset = LocalizationDataset(X_features, y_target)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = CompactResidualTransformer()
    criterion = nn.MSELoss()
    # AdamW 옵티마이저 (Weight Decay로 과적합 방지)
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4) 
    
    epochs = 200
    best_val_loss = float('inf')
    
    print("--- 🚀 GERT Transformer 학습 시작 ---")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= train_size
        
        # 검증(Validation) 세트 평가
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= val_size
        
        if (epoch+1) % 40 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss(MSE): {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        # 최고 성능 모델 저장 (Early Stopping 효과)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model.pt')
            
    print(f"✅ 학습 완료! 최고 검증 성능 기준 저장됨 (최저 오차: {best_val_loss:.4f})")
    print("💾 'model.pt' 파일이 동일 폴더에 생성되었습니다.")

if __name__ == "__main__":
    train_model()