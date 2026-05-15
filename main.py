import numpy as np
import scipy.io as sio  
import torch
import torch.nn as nn

# =======================================================
# 1. AI 모델 구조 (훈련 시와 100% 동일)
# =======================================================
class CompactResidualTransformer(nn.Module):
    def __init__(self, num_anchors=18, feature_dim=2, d_model=32, nhead=4, num_layers=1):
        super(CompactResidualTransformer, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_anchors, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, 
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 2)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding 
        attn_out = self.transformer(x)
        pooled_out = torch.mean(attn_out, dim=1)
        return self.fc_out(pooled_out)

# =======================================================
# 2. GERT 전처리 함수
# =======================================================
def get_gert_features(d_hat, p_bs):
    # p_bs: (2, 18) -> (18, 2)
    bs_pos = p_bs.T
    
    # 1. 초기 위치 추정 (가중 평균)
    inv_d = 1.0 / (d_hat + 1e-6)
    weights = inv_d / np.sum(inv_d, axis=0)
    initial_guesses = (weights.T @ bs_pos)
    
    # 2. 기하학적 잔차(Residual) 추출
    num_user = d_hat.shape[1]
    residual_features = np.zeros((num_user, 18))
    for i in range(num_user):
        guess_pos = initial_guesses[i]
        physical_dist = np.linalg.norm(bs_pos - guess_pos, axis=1)
        residual_features[i, :] = d_hat[:, i] - physical_dist
        
    # 3. Transformer 입력용 (N, 18, 2) 텐서 변환
    X_seq = np.stack([d_hat.T, residual_features], axis=-1)
    return torch.tensor(X_seq, dtype=torch.float32), initial_guesses

# =======================================================
# 3. 메인 함수 
# =======================================================
def main():
    import os
    mat_path = 'DH_FR1.mat'
    if not os.path.exists(mat_path):
        mat_path = 'InF_DH_FR1.mat'
    
    data = sio.loadmat(mat_path, squeeze_me=False)
    
    # [KeyError 해결] 'p_bs'가 없으면 'BS_positions'를 가져옴
    if 'p_bs' in data:
        p_bs = np.asarray(data['p_bs'], dtype=float)
    else:
        p_bs = np.asarray(data['BS_positions'], dtype=float)
        
    d_hat = np.asarray(data['d_hat'], dtype=float)
    num_user = d_hat.shape[1]
    
    # --- 알고리즘 실행 ---
    # 1. 전처리
    X_tensor, initial_guesses = get_gert_features(d_hat, p_bs)
    
    # 2. 모델 로드
    model = CompactResidualTransformer()
    try:
        # [중요 수정] 최신 PyTorch 2.8.0 호환성을 위한 weights_only=True 추가
        model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        
        # 3. AI 보정값 예측
        with torch.no_grad():
            predicted_residuals = model(X_tensor).numpy() 
            
        # 4. 최종 위치 = 초기 추측 + AI 보정
        p_hat_processed = initial_guesses + predicted_residuals
    except Exception as e:
        # 모델 로드 실패 시 가중평균값(Baseline)이라도 안전하게 반환
        print(f"⚠️ 모델 로드 실패, 초기 가중 평균값으로 대체합니다. (에러: {e})")
        p_hat_processed = initial_guesses
    
    # 규칙: (2, num_user) 형태로 반환
    p_hat = p_hat_processed.T 
    
    return p_hat

if __name__ == "__main__":
    estimated_p = main()
    print(f"✅ 측위 완료! 결과 형태: {estimated_p.shape}")