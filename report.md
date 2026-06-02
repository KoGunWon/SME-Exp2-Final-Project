# Geo-Explainable Residual Transformer for Robust WiFi RTT Indoor Localization

| 항목 | 내용 |
|---|---|
| 이름 | 고건원 |
| 학번 | 12223621 |
| 프로젝트 | 스마트모빌리티공학실험2 Final Project |
| 제출 파일 | main.py, train.py, model.pt, report.md |

## 1. 모티베이션 & 인트로

WiFi RTT 기반 실내 측위는 사용자와 여러 기지국 또는 앵커 사이의 거리 추정값을 이용하여 사용자의 2차원 위치를 추정하는 range-based localization 문제이다. 이상적인 LOS 환경에서는 각 기지국을 중심으로 하는 거리 원들이 사용자 위치 근처에서 교차하므로, 삼변측량 또는 비선형 least-squares 형태로 위치를 계산할 수 있다. 그러나 실제 실내 환경에서는 벽, 기둥, 금속 구조물, 사람의 이동, 반사체 등에 의해 NLOS와 multipath가 발생한다. 이 경우 측정 거리 \(\hat{d}_i\)는 실제 거리 \(\|p-p_{bs,i}\|_2\)와 일치하지 않고, 특정 기지국에서 큰 양의 bias를 가지거나 일부 샘플에서 극단적으로 튀는 값으로 나타난다. 따라서 단순 삼변측량이나 단순 평균 기반 위치 추정은 일부 오염된 거리값에 크게 끌려갈 수 있다.

중간발표 전후의 실험에서는 먼저 통계 기반 강건화 방법을 검토하였다. MAD 기반 이상치 처리, DBSCAN 클러스터링, MCC 기반 강건 추정과 같은 방법은 튀는 거리값을 일부 완화할 수 있었다. 그러나 이러한 방법은 대부분 개별 거리값 또는 샘플 집합의 통계적 분포를 중심으로 판단한다. 실내 측위에서는 단일 기지국의 거리값이 통계적으로 정상 범위에 있더라도 다른 17개 기지국과 함께 보면 기하학적으로 모순될 수 있다. 반대로 통계적으로는 큰 거리값처럼 보여도 실제 사용자가 해당 기지국에서 멀리 위치한다면 정상적인 측정일 수 있다. 즉, WiFi RTT 위치추정에서 중요한 것은 단순 outlier 여부가 아니라, 18개 앵커가 동시에 관측한 거리 패턴이 하나의 2차원 위치와 얼마나 일관되는가이다.

이 한계에서 출발하여 본 프로젝트에서는 순수 통계 필터 또는 순수 딥러닝 대신, 기하학 기반 초기 위치 추정과 머신러닝 기반 residual 보정을 결합한 GERT, Geo-Explainable Residual Transformer를 설계하였다. 핵심 아이디어는 인공지능이 사용자 좌표를 처음부터 직접 예측하지 않도록 제한하는 것이다. 먼저 거리값으로 물리적으로 그럴듯한 초기 위치를 만들고, 그 초기 위치가 실제 위치로 이동하기 위해 필요한 보정량만 학습하게 한다. 이렇게 하면 모델이 2차원 좌표 공간 전체를 암기하는 부담이 줄어들고, WiFi RTT 거리 방정식에서 발생하는 기하학적 모순을 중심으로 학습하게 된다.

본 프로젝트에서 제안하는 구조는 “black-box 좌표 회귀”가 아니라 “physics-guided residual regression”에 가깝다. 거리값은 물리적으로 사용자와 기지국 사이의 range measurement를 의미하므로 완전히 무시할 수 없다. 다만 실내 NLOS 환경에서는 이 거리값이 신뢰할 수 없는 경우가 많다. 따라서 본 알고리즘은 거리값으로부터 초기 위치를 만들되, 그 초기 위치가 각 기지국의 거리 방정식과 얼마나 어긋나는지를 residual feature로 계산한다. 이후 경량 Transformer가 18개 기지국 residual의 상호 패턴을 학습하여 최종 보정 벡터를 출력한다.

본 프로젝트의 기여는 다음과 같이 정리할 수 있다.

| 기여 | 내용 | 설계 의도 |
|---|---|---|
| 물리 기반 초기화 | inverse-distance weighted centroid로 초기 위치를 만든다. | 모든 입력에서 안정적으로 계산되고, 거리값의 물리적 의미를 보존한다. |
| 기하학적 residual feature | 측정 거리와 초기 위치 기준 물리 거리의 차이를 계산한다. | 단순 거리 크기보다 “거리 방정식의 모순”을 모델에 직접 제공한다. |
| Compact Transformer 보정기 | 18개 기지국을 token으로 보고 residual correction을 수행한다. | 앵커 간 상관관계를 학습하되, 700개 데이터에 맞게 모델 크기를 제한한다. |
| residual regression | 절대 좌표가 아니라 \(p_{true}-\hat{p}_{init}\)만 예측한다. | 좌표 직접 예측보다 과적합 위험을 줄인다. |
| 제출 안정성 | main.py에서 전체 사용자 batch 추론을 수행하고 model.pt를 로드한다. | hidden test와 실행 시간 제한에 안정적으로 대응한다. |

본 프로젝트에서 사용한 데이터와 기본 설정은 다음과 같다. 제공 데이터 700개 중 595개는 학습에, 105개는 검증에 사용하였다. 최종 제출용 model.pt는 검증 실험 이후 전체 700개 샘플로 재학습하여 저장하였다.

| 항목 | 값 |
|---|---:|
| 전체 제공 샘플 수 | 700 |
| 기지국 수 | 18 |
| 거리 입력 행렬 | 18 × 700 |
| 기지국 좌표 행렬 | 2 × 18 |
| 정답 위치 행렬 | 2 × 700 |
| 학습 샘플 수 | 595 |
| 검증 샘플 수 | 105 |
| 검증 비율 | 0.15 |
| 최종 제출 모델 파일 | model.pt |

전체 알고리즘은 먼저 MAD 기반 clipping으로 극단적인 거리값을 완화하고, inverse-distance weighted centroid로 초기 위치를 계산한다. 이후 각 기지국에서 측정된 거리와 초기 위치 기준 물리적 거리의 차이를 residual feature로 만들고, 18개 기지국을 Transformer의 token으로 취급하여 보정 벡터를 예측한다. 최종 위치는 초기 위치에 보정 벡터를 더해 계산된다. 이 구조는 성능만을 목표로 한 단순 딥러닝이 아니라, 통신·측위 문제의 물리적 구조를 보존하면서 AI가 보정해야 할 부분을 제한한 하이브리드 측위 알고리즘이다.

## 2. 알고리즘 설명

### 2.1 문제 표기와 데이터 구조

사용자 위치를 \(p=(x,y)\in\mathbb{R}^2\), \(i\)번째 기지국 좌표를 \(p_{bs,i}\in\mathbb{R}^2\)라 둔다. 기지국 수는 \(M=18\)이고, 사용자 수는 \(N\)이다. 사용자 \(u\)에 대해 측정된 WiFi RTT 거리 벡터는 다음과 같다.

$$
\hat{d}_u=[\hat{d}_{1,u},\hat{d}_{2,u},\cdots,\hat{d}_{18,u}]^T
$$

이상적인 거리 모델은 다음과 같이 쓸 수 있다.

$$
\hat{d}_{i,u}=\|p_u-p_{bs,i}\|_2+\eta_{i,u}
$$

여기서 \(\eta_{i,u}\)는 측정 오차이다. LOS 환경에서는 \(\eta_{i,u}\)가 작고 평균이 0에 가까운 잡음으로 볼 수 있다. 그러나 NLOS 환경에서는 특정 기지국에서 양의 bias가 누적되거나, 반사 경로 때문에 실제 거리보다 긴 값이 관측될 수 있다. 따라서 본 문제는 단순히 18개 거리값을 모두 동일한 신뢰도로 맞추는 문제가 아니다. 어떤 기지국의 거리값이 현재 기하 구조와 일관되는지, 어떤 기지국의 거리값이 전체 거리 패턴과 충돌하는지를 판단해야 한다.

채점에서 요구하는 최종 출력은 모든 사용자에 대한 예측 위치 행렬이다.

$$
\hat{P}=\begin{bmatrix}
\hat{x}_1 & \hat{x}_2 & \cdots & \hat{x}_N \\
\hat{y}_1 & \hat{y}_2 & \cdots & \hat{y}_N
\end{bmatrix}
$$

따라서 main.py는 최종적으로 \((2,N)\) 형태의 numpy 배열을 반환한다. 내부 학습과 추론에서는 계산 편의를 위해 위치 행렬을 \((N,2)\) 형태로 다루고, 반환 직전에 transpose한다. 이 설계는 사용자 수를 700명으로 고정하지 않고, hidden test의 \(N=300\) 같은 임의의 사용자 수에도 대응할 수 있도록 한다.

### 2.2 기존 삼변측량 관점과 본 알고리즘의 위치

전통적인 range-based localization은 다음의 비선형 least-squares 문제로 나타낼 수 있다.

$$
\hat{p}=\arg\min_p\sum_{i=1}^{18}\left(\|p-p_{bs,i}\|_2-\hat{d}_i\right)^2
$$

이 목적함수는 모든 기지국의 거리 오차를 제곱합으로 줄이는 위치를 찾는다. 하지만 NLOS 거리값이 섞여 있으면 제곱 손실은 큰 오차에 민감하기 때문에 오염된 기지국이 전체 위치를 크게 왜곡할 수 있다. Weighted least-squares나 robust least-squares를 사용하면 이 문제를 완화할 수 있지만, 기지국별 신뢰도 추정, 초기값, 반복 최적화 안정성 등의 문제가 남는다.

본 프로젝트는 엄밀한 least-squares 삼변측량을 직접 구현하지 않았다. 대신 삼변측량의 핵심인 range measurement의 물리적 의미를 초기 위치 추정에 사용하고, 남은 오차 구조를 AI가 residual 형태로 보정하도록 만들었다. 이 선택은 두 가지 이유에서 이루어졌다. 첫째, 과제의 채점 환경에서는 main.py가 hidden 데이터에 대해 안정적으로 실행되어야 하므로 반복 최적화 실패 가능성을 줄이는 것이 중요하다. 둘째, 제공 데이터가 700개로 제한되어 있으므로 큰 신경망이 좌표를 직접 외우는 구조보다, 물리 기반 초기값을 두고 보정량만 학습하는 구조가 더 적합하다.

따라서 GERT는 다음과 같은 중간 위치에 있다.

| 접근 | 특징 | 본 프로젝트와의 관계 |
|---|---|---|
| 순수 삼변측량 | 거리 방정식을 직접 풀어 위치를 계산한다. | 물리적 해석성은 좋지만 NLOS에 민감할 수 있다. |
| 순수 딥러닝 좌표 회귀 | 거리값에서 \((x,y)\)를 직접 예측한다. | 데이터가 적으면 좌표 분포를 외울 위험이 있다. |
| GERT | 초기 위치는 물리 기반으로 만들고, AI는 residual만 보정한다. | 물리 구조와 데이터 기반 보정을 결합한다. |

### 2.3 MAD 기반 거리값 정제

실내 RTT 데이터에서는 특정 기지국의 거리값이 일부 사용자 샘플에서 비정상적으로 커질 수 있다. 단순 평균과 표준편차는 이런 극단값에 민감하므로, 본 프로젝트에서는 median과 MAD를 사용하였다. 기지국 \(i\)에 대해 전체 사용자 거리값 집합을 \(D_i=\{\hat{d}_{i,1},\ldots,\hat{d}_{i,N}\}\)라 하면, median과 MAD는 다음과 같이 정의된다.

$$
\text{med}_i = \text{median}(D_i)
$$

$$
\text{MAD}_i = \text{median}\left(|\hat{d}_{i,u}-\text{med}_i|\right)
$$

Modified Z-score는 다음과 같이 계산하였다.

$$
z_{i,u}=0.6745\cdot \frac{\hat{d}_{i,u}-\text{med}_i}{\text{MAD}_i+\epsilon}
$$

여기서 0.6745는 정규분포에서 MAD를 표준편차와 유사한 scale로 맞추기 위해 사용되는 상수이고, \(\epsilon\)은 MAD가 0이 되는 경우를 피하기 위한 작은 값이다. 본 프로젝트에서는 \(|z_{i,u}|>3.5\)인 값을 이상치 후보로 보고 clipping하였다.

| 항목 | 설정 | 선정 이유 |
|---|---:|---|
| 기준 통계량 | median, MAD | 평균과 표준편차보다 극단적 RTT 오차에 덜 민감하다. |
| Modified Z-score threshold | 3.5 | robust statistics에서 흔히 쓰이는 강한 이상치 기준이며, 정상 거리값을 과도하게 삭제하지 않는다. |
| 처리 방식 | 제거가 아니라 clipping | Transformer 입력 token 수 18개를 유지해야 하므로 거리값을 삭제하지 않는다. |
| clipping 범위 | 해당 기지국의 정상 샘플 최소값과 최대값 | 데이터 shape를 유지하면서 극단값의 영향만 제한한다. |

이상치를 삭제하지 않고 clipping한 이유가 중요하다. Transformer는 각 사용자마다 18개 기지국 token이 모두 존재한다고 가정한다. 특정 기지국을 삭제하면 사용자마다 token 수가 달라지고, 모델 입력 shape가 불안정해진다. 또한 hidden test에서 어떤 기지국의 측정값이 튀었는지 모르기 때문에, 특정 기지국을 완전히 버리는 방식은 위험하다. 따라서 본 프로젝트에서는 “정보 삭제”보다 “극단값 영향 제한”을 선택하였다.

MAD clipping은 최종 성능을 직접 만들어내는 주된 AI 요소는 아니다. 이 단계의 목적은 모델 입력의 동적 범위를 안정화하고, 극단적인 RTT 값이 초기 위치 계산과 residual feature를 동시에 망가뜨리는 것을 막는 것이다. 즉, MAD clipping은 학습 모델의 입력 분포를 안정화하는 전처리이며, 실제 위치 보정의 핵심은 이후 residual regression에서 수행된다.

### 2.4 거리 역수 가중 중심 기반 초기 위치

MAD clipping 후에는 inverse-distance weighted centroid를 사용하여 초기 위치 \(\hat{p}_{init}\)를 계산한다. 거리 역수 가중치는 다음과 같다.

$$
\tilde{w}_i=\frac{1}{\hat{d}_i+\epsilon}
$$

$$
w_i=\frac{\tilde{w}_i}{\sum_{j=1}^{18}\tilde{w}_j}
$$

초기 위치는 다음과 같이 정의된다.

$$
\hat{p}_{init}=\sum_{i=1}^{18}w_i p_{bs,i}
$$

이 수식은 “측정 거리가 짧은 기지국일수록 사용자의 위치에 더 가까울 가능성이 높다”는 직관을 반영한다. 물론 실내 NLOS 환경에서는 가까운 기지국의 거리도 왜곡될 수 있다. 그러나 전체적으로는 거리값이 사용자의 위치에 대한 coarse geometric cue를 제공하므로, weighted centroid는 완전히 무작위적인 초기점보다 훨씬 유리하다.

초기 위치 추정기의 역할은 최종 정답을 완벽히 맞히는 것이 아니다. GERT에서 초기 위치는 residual feature를 만들기 위한 기준점이다. 초기 위치가 주어지면 각 기지국까지의 물리 거리 \(\rho_i\)를 계산할 수 있고, 측정 거리 \(\hat{d}_i\)와 비교하여 기하학적 모순 \(r_i\)를 만들 수 있다. 따라서 초기 위치는 “정답 후보”이면서 동시에 “거리 방정식 검사용 기준점”이다.

초기 추정기로 weighted centroid를 선택한 이유는 다음과 같다.

| 요구 조건 | weighted centroid가 적합한 이유 |
|---|---|
| 계산 안정성 | 역행렬 계산이나 반복 최적화가 없어 모든 입력에서 안정적으로 계산된다. |
| 실행 속도 | 모든 사용자에 대해 행렬 곱으로 빠르게 계산할 수 있다. |
| 물리적 의미 | 짧은 거리값을 더 신뢰한다는 range-based 직관을 반영한다. |
| residual 생성 가능성 | 초기 위치 기준 물리 거리를 계산할 수 있어 residual feature를 만들 수 있다. |
| hidden test 대응 | 데이터 크기와 사용자 수가 바뀌어도 동일한 방식으로 동작한다. |

### 2.5 기하학적 residual feature

초기 위치 \(\hat{p}_{init}\)가 계산되면, 각 기지국에 대해 초기 위치 기준 물리적 거리와 실제 측정 거리의 차이를 계산한다.

$$
\rho_i=\|\hat{p}_{init}-p_{bs,i}\|_2
$$

$$
r_i=\hat{d}_i-\rho_i
$$

여기서 \(\rho_i\)는 초기 위치가 맞다고 가정했을 때 \(i\)번째 기지국까지의 물리적 거리이고, \(r_i\)는 측정 거리와 물리 거리 사이의 residual이다. 이 residual은 단순한 오차값이 아니라, “현재 초기 위치가 각 기지국의 거리 방정식을 얼마나 만족하지 못하는가”를 나타내는 기하학적 정보이다.

Residual의 부호와 크기는 다음과 같이 해석할 수 있다.

| residual 형태 | 해석 |
|---|---|
| \(r_i\approx 0\) | 측정 거리와 초기 위치 기준 물리 거리가 잘 맞는다. |
| \(r_i>0\) | 측정 거리가 초기 위치 기준 거리보다 크다. NLOS 또는 양의 bias 가능성이 있다. |
| \(r_i<0\) | 측정 거리가 초기 위치 기준 거리보다 작다. 초기 위치가 해당 기지국에서 너무 멀리 잡혔을 수 있다. |
| 특정 기지국만 큰 \(|r_i|\) | 그 기지국 측정값이 다른 기지국 조합과 기하학적으로 불일치할 가능성이 있다. |

최종 Transformer 입력은 각 기지국을 하나의 token으로 보고, token feature를 측정 거리와 residual의 쌍으로 구성한다.

$$
X_i=[\hat{d}_i, r_i]
$$

사용자 한 명의 입력 행렬은 다음과 같다.

$$
X=\begin{bmatrix}
\hat{d}_1 & r_1 \\
\hat{d}_2 & r_2 \\
\vdots & \vdots \\
\hat{d}_{18} & r_{18}
\end{bmatrix}\in\mathbb{R}^{18\times 2}
$$

이 설계가 본 프로젝트의 핵심이다. 거리값만 입력하면 모델은 \(\hat{d}_i\)와 위치 보정량 사이의 관계를 데이터에서 직접 찾아야 한다. 그러나 residual을 함께 주면 모델은 이미 “측정값이 현재 기하 구조와 얼마나 모순되는가”라는 물리적 힌트를 받는다. 따라서 제한된 데이터 환경에서도 모델이 더 빠르게 의미 있는 패턴을 학습할 수 있다.

Residual feature는 다음 세 가지 측면에서 유리하다.

| 관점 | 설명 |
|---|---|
| 물리 해석성 | 거리 방정식의 불일치를 직접 표현한다. |
| 학습 효율 | 모델이 원시 거리값만으로 기하학을 추론해야 하는 부담을 줄인다. |
| 일반화 가능성 | 좌표 자체보다 기하학적 모순 패턴은 hidden test에서도 유지될 가능성이 높다. |

### 2.6 Compact Residual Transformer 구조

18개 기지국은 고정된 순서를 가지는 측위 센서 집합이다. 본 프로젝트에서는 각 기지국을 자연어 문장의 단어처럼 하나의 token으로 보았다. Transformer Encoder는 token 사이의 상호작용을 학습할 수 있으므로, 특정 기지국 하나만 독립적으로 보는 MLP보다 앵커 간 상대적 패턴을 표현하기에 적합하다.

각 token \(X_i\in\mathbb{R}^2\)는 선형 embedding을 거쳐 \(d_{model}\)차원 벡터가 된다.

$$
h_i^{(0)}=W_eX_i+b_e+p_i
$$

여기서 \(p_i\)는 \(i\)번째 기지국 token의 positional embedding이다. 이 embedding은 기지국 순서 또는 식별 정보를 모델이 구분할 수 있도록 한다. 모든 기지국이 같은 선형 embedding을 통과하더라도, positional embedding이 더해지면 모델은 “몇 번째 기지국의 거리와 residual인가”를 구분할 수 있다.

Self-attention의 기본 구조는 다음과 같다.

$$
Q=HW_Q,\quad K=HW_K,\quad V=HW_V
$$

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

이 구조를 통해 모델은 어떤 기지국 token이 다른 기지국 token과 함께 볼 때 중요한지를 학습한다. 예를 들어 어떤 기지국의 residual이 큰데 주변 기지국들과 거리 패턴이 맞지 않는다면, attention 구조는 그 token의 영향을 조절하는 방향의 표현을 만들 수 있다. 본 프로젝트에서는 attention weight를 별도로 시각화하지 않았으므로 “모델이 실제로 특정 기지국을 억제했다”고 단정하지 않는다. 다만 구조적으로 앵커 간 상관관계를 학습할 수 있도록 설계했다는 점이 중요하다.

Transformer Encoder의 출력은 18개 token에 대한 hidden representation이다. 이를 평균 pooling하여 전체 사용자 샘플을 대표하는 벡터로 만든다.

$$
h_{pool}=\frac{1}{18}\sum_{i=1}^{18}h_i
$$

마지막 출력 head는 2차원 보정 벡터를 출력한다.

$$
\Delta\hat{p}=f_\theta(X)=[\Delta\hat{x},\Delta\hat{y}]
$$

최종 위치는 다음과 같다.

$$
\hat{p}_{final}=\hat{p}_{init}+\Delta\hat{p}
$$

학습 label은 실제 위치와 초기 위치의 차이로 정의하였다.

$$
y=p_{true}-\hat{p}_{init}
$$

손실 함수는 보정 벡터에 대한 MSE이다.

$$
\mathcal{L}(\theta)=\frac{1}{N}\sum_{u=1}^{N}\left\|f_\theta(X_u)-(p_{true,u}-\hat{p}_{init,u})\right\|_2^2
$$

이 구조는 좌표 직접 예측과 다르다. 좌표 직접 예측에서는 모델이 \((x,y)\) 전체를 출력해야 하므로 데이터의 공간 분포를 외우기 쉽다. 반면 residual regression에서는 모델이 물리 기반 초기 위치의 오차만 보정하므로 출력 범위가 상대적으로 제한된다. 이 점이 제한된 700개 데이터 환경에서 중요한 일반화 전략이다.

### 2.7 모델 파라미터와 선정 이유

최종 모델은 일부러 크게 만들지 않았다. 제공 데이터가 700개뿐이고 validation에 실제로 사용하는 학습 샘플은 595개이므로, 지나치게 큰 Transformer는 validation set에 쉽게 과적합될 수 있다. 따라서 1-layer encoder, 32차원 embedding, 4-head attention으로 제한하였다.

| 구성 요소 | 최종 설정 | 선정 이유 |
|---|---:|---|
| 입력 token 수 | 18 | 기지국 수와 동일하다. |
| token feature 수 | 2 | 거리와 residual을 함께 사용한다. |
| embedding dimension | 32 | 2차원 feature를 충분히 표현하되, 데이터 수 대비 과도하게 크지 않다. |
| attention head 수 | 4 | 32차원을 4개 head로 나누면 head당 8차원으로 안정적이다. |
| Transformer layer 수 | 1 | 앵커 수가 18개로 작고 데이터도 적으므로 깊은 구조를 피했다. |
| feedforward dimension | 64 | embedding dimension의 2배 수준으로 비선형 표현력을 부여한다. |
| encoder dropout | 0.2 | 작은 데이터에서 과적합을 줄이기 위한 정규화이다. |
| output head dropout | 0.1 | 최종 보정량 회귀에서 과도한 co-adaptation을 줄인다. |
| optimizer | AdamW | weight decay를 분리하여 작은 모델의 일반화 안정성을 높인다. |
| learning rate | 0.005 | 200 epoch 안에서 충분히 빠르게 수렴하면서 발산하지 않는 값으로 선택했다. |
| weight decay | 0.0001 | 작은 L2 계열 정규화로 과적합을 완화한다. |
| trainable parameters | 9,778 | 과제 데이터 크기에 비해 과도하지 않은 경량 모델이다. |

모델 크기를 제한한 이유는 단순히 실행 시간을 줄이기 위해서만은 아니다. 실내 측위 데이터는 좌표 분포가 제한되어 있고, hidden test 300개가 제공 데이터 700개와 완전히 같은 분포라고 보장할 수 없다. 큰 모델은 제공 데이터의 좌표 배치를 암기하는 방향으로 학습될 수 있다. 반면 작은 모델은 표현력이 제한되므로, 물리 기반 feature가 제공하는 구조적 정보에 더 의존하게 된다. 본 프로젝트에서는 residual feature를 강하게 설계하고 모델은 경량화하여, “feature engineering은 물리적으로 풍부하게, 모델은 compact하게”라는 원칙을 적용하였다.

학습률 0.005는 다소 큰 값처럼 보일 수 있지만, 모델이 9,778개 파라미터로 작고 학습 target이 2차원 residual이기 때문에 빠른 수렴에 유리했다. 반대로 학습률을 너무 낮추면 200 epoch 안에서 충분히 수렴하지 않을 수 있다. weight decay는 매우 크게 주지 않았다. 이미 dropout과 작은 모델 크기가 정규화 역할을 하므로, 큰 weight decay를 적용하면 보정 벡터의 표현력이 부족해질 수 있기 때문이다.

### 2.8 학습과 제출용 모델 생성

학습 과정은 두 단계로 나누었다. 첫 번째 단계에서는 700개 제공 데이터 중 85%를 train set, 15%를 validation set으로 분리하였다. 이 검증 실험은 모델이 학습에 사용하지 않은 샘플에서 얼마나 잘 동작하는지 확인하기 위한 것이다. 두 번째 단계에서는 최종 제출용 model.pt를 만들기 위해 전체 700개 샘플로 다시 학습하였다.

이 두 단계를 분리한 이유는 평가 해석의 공정성 때문이다. Validation 결과는 일반화 가능성을 판단하기 위한 값이고, 전체 데이터 재학습 결과는 최종 model.pt가 학습 데이터에 대해 정상적으로 residual 보정을 수행하는지 확인하는 sanity check이다. 따라서 전체 데이터로 재학습한 뒤 같은 데이터에서 계산한 성능을 hidden test 성능으로 주장하지 않았다.

| 단계 | 목적 | 사용 데이터 | 보고서에서의 의미 |
|---|---|---|---|
| Validation training | 일반화 가능성 평가 | train 595, validation 105 | 주요 성능 평가 |
| Final training | 제출용 model.pt 생성 | 전체 700 | 최종 모델 sanity check |

제출용 main.py에서는 학습 때와 동일한 핵심 전처리를 적용한다. 즉, shape 정규화, MAD clipping, weighted centroid 초기 위치, residual feature 생성, Transformer 추론, 최종 위치 반환 흐름이 train.py와 일치한다. 이 일치성이 중요하다. 학습 때 사용한 입력 분포와 추론 때 사용한 입력 분포가 달라지면 model.pt의 보정량이 불안정해질 수 있기 때문이다.

### 2.9 파라미터 민감도 및 보조 ablation 실험

최종 제출 모델은 200 epoch 학습으로 만들었지만, 파라미터 선정의 타당성을 보기 위해 동일 train/validation split에서 30 epoch 보조 민감도 실험도 수행하였다. 이 실험은 최종 성능을 주장하기 위한 것이 아니라, residual feature와 모델 크기, learning rate 변화가 어떤 경향을 보이는지 확인하기 위한 보조 분석이다.

| 실험 | 변경 사항 | 파라미터 수 | Mean | RMSE | P90 | Max | 해석 |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline | weighted centroid only | 0 | 23.86 | 25.54 | 35.58 | 43.84 | AI 보정 전 기준선 |
| Full GERT 30ep | d_model 32, residual 사용 | 9,778 | 12.92 | 14.70 | 22.94 | 27.89 | 짧은 학습만으로도 baseline보다 크게 개선 |
| Distance-only 30ep | residual channel 제거 | 9,778 | 15.66 | 17.27 | 25.28 | 31.37 | residual을 제거하면 성능이 하락 |
| Small model 30ep | d_model 16 | 3,922 | 16.36 | 18.16 | 26.40 | 36.82 | 모델이 너무 작으면 보정 표현력이 부족 |
| Large model 30ep | d_model 64 | 27,634 | 13.34 | 15.06 | 22.74 | 32.28 | 파라미터가 커져도 단기 검증 성능이 크게 좋아지지 않음 |
| Low LR 30ep | learning rate 0.001 | 9,778 | 12.91 | 14.85 | 22.29 | 30.72 | 안정적이나 빠른 개선에는 기본 LR과 큰 차이 없음 |

이 결과에서 가장 중요한 비교는 Full GERT와 Distance-only이다. 같은 모델 구조를 사용하되 residual channel만 제거했을 때 Mean error가 12.92 m에서 15.66 m로 악화되었다. 이는 단순 거리값보다 기하학적 residual이 보정 문제에 중요한 정보를 제공한다는 것을 뒷받침한다. 또한 d_model 16은 파라미터 수는 적지만 성능이 나빠졌고, d_model 64는 파라미터 수가 약 2.8배 증가했음에도 30 epoch 기준 Full GERT보다 뚜렷한 개선을 보이지 않았다. 따라서 최종 설정인 d_model 32는 표현력과 과적합 위험 사이의 균형점으로 판단하였다.

해당 민감도 실험은 길게 학습한 최종 모델의 절대 성능을 대체하지 않는다. 그러나 설계 판단의 방향성을 뒷받침한다. 첫째, residual feature가 실제로 정보량을 가진다. 둘째, 모델을 무작정 크게 만드는 것이 항상 좋은 결과로 이어지지 않는다. 셋째, 학습률은 너무 낮아도 단기 수렴이 느려질 수 있고, 기본값 0.005는 작은 모델에서는 합리적인 선택이었다.

### 2.10 관련 연구와의 차별점 요약

본 프로젝트는 기존 연구의 개념을 그대로 구현한 것이 아니라, 여러 문헌에서 필요한 원리를 참고하고 과제 데이터의 제약에 맞게 결합한 것이다. 실내 측위 연구에서 range-based localization과 NLOS 문제가 중요하다는 점은 기존 survey와 UWB localization 문헌에서 참고하였다. Robust scale estimation은 MAD 기반 clipping 설계의 통계적 근거가 되었다. Transformer는 앵커 token 간 관계를 학습하는 구조적 아이디어를 제공하였다. Physics-informed learning은 물리 법칙과 학습 모델을 결합한다는 큰 방향성에서 참고하였다.

그러나 본 프로젝트의 핵심 구현은 기존 논문과 다르다. 본 알고리즘은 WiFi RTT 데이터에서 inverse-distance weighted centroid로 초기 위치를 만들고, 측정 거리와 초기 위치 기반 물리 거리의 차이를 residual feature로 구성한 뒤, 18개 기지국을 token으로 하는 compact Transformer가 보정 벡터를 예측한다. 이 조합은 본 과제의 데이터 크기, 실행 시간 제한, hidden test 조건을 고려하여 직접 설계한 것이다.

## 3. Agent AI 활용 방안

본 프로젝트에서는 Agent AI를 알고리즘 설계의 대체자가 아니라 구현 보조 및 검토 도구로 사용하였다. 핵심 알고리즘의 방향, 즉 “거리 기반 초기 위치를 먼저 계산하고, 딥러닝은 위치 자체가 아니라 residual 보정량만 학습하게 한다”는 판단은 WiFi RTT 데이터의 물리적 특성과 과적합 위험을 고려하여 직접 결정하였다.

본인이 주도한 부분은 다음과 같다.

| 구분 | 본인이 수행한 역할 |
|---|---|
| 문제 정의 | WiFi RTT 실내 측위에서 NLOS와 multipath가 핵심 오차 요인임을 분석하였다. |
| 중간발표 이후 방향 설정 | 단순 통계 필터만으로는 18개 앵커의 기하 관계를 충분히 활용하기 어렵다고 판단하였다. |
| 알고리즘 구조 | 좌표 직접 예측 대신 초기 위치 기반 residual regression 구조를 선택하였다. |
| feature 설계 | 거리와 기하학적 residual을 함께 사용하는 token feature를 설계하였다. |
| 모델 선택 | 18개 기지국을 token으로 보는 Compact Transformer 구조를 선택하였다. |
| 파라미터 결정 | 데이터 수와 실행 시간 제한을 고려하여 1-layer, d_model 32, 4-head 구조를 선택하였다. |
| 평가 해석 | validation 성능과 all-data sanity check를 구분하여 해석하였다. |
| 제출 구성 | main.py, train.py, model.pt, report.md의 역할을 분리하였다. |

Agent AI는 다음 업무에 보조적으로 활용하였다.

| 구분 | Agent AI 활용 내용 |
|---|---|
| 코드 구조화 | PyTorch 기반 Compact Residual Transformer 구현 구조를 점검하였다. |
| 디버깅 | d_hat, BS_positions, p의 shape 불일치 문제를 확인하고, (18,N), (2,18), (N,2) 변환 규칙을 정리하였다. |
| 학습 코드 점검 | train.py와 main.py의 전처리 순서가 일치하는지 확인하였다. |
| 모델 저장 | model.pt 저장 및 main.py에서의 로딩 흐름을 검토하였다. |
| 결과 정리 | Mean, Median, RMSE, P90, Max error를 markdown 표로 정리하였다. |
| 보고서 점검 | 과장된 표현을 줄이고, 실제 구현과 설명이 어긋나지 않도록 문장을 수정하였다. |
| Reference 보강 | 단순 문헌 목록이 아니라, 각 문헌에서 무엇을 참고했고 본 프로젝트에서 무엇을 직접 설계했는지 분리하여 서술하도록 보조하였다. |

Agent AI가 제안한 내용은 그대로 채택하지 않고, 실제 코드 실행 결과와 제출 규격에 맞는지 확인한 뒤 반영하였다. 예를 들어 attention 구조는 앵커 간 상관관계를 학습할 수 있는 구조라는 점에서 설명하되, attention weight를 직접 시각화하지 않았으므로 특정 앵커를 실제로 억제했다고 단정하지 않았다. 또한 전체 700개 데이터로 재학습한 모델의 성능은 hidden test 성능이 아니라 sanity check라고 명확히 구분하였다.

## 4. 결과 도출 & 디스커션

### 4.1 자체 평가 방식과 metric 정의

전체 700개 제공 데이터를 seed 42 기준으로 무작위 분리하여 595개 train set과 105개 validation set을 만들었다. 모델은 train set만 사용해 학습하고, validation set은 성능 평가에만 사용하였다. 최종 제출용 model.pt는 검증 실험 이후 전체 700개 샘플을 이용해 다시 학습하였다.

| 평가 설정 | 값 |
|---|---:|
| seed | 42 |
| train samples | 595 |
| validation samples | 105 |
| epochs | 200 |
| batch size | 32 |
| optimizer | AdamW |
| learning rate | 0.005 |
| weight decay | 0.0001 |
| best validation MSE | 54.60 |

위치 오차는 예측 위치와 정답 위치 사이의 유클리드 거리로 정의하였다.

$$
e_u=\sqrt{(\hat{x}_u-x_u)^2+(\hat{y}_u-y_u)^2}
$$

Mean error는 전체 평균 오차이다.

$$
\text{Mean Error}=\frac{1}{N}\sum_{u=1}^{N}e_u
$$

RMSE는 큰 오차에 더 민감한 지표이다.

$$
\text{RMSE}=\sqrt{\frac{1}{N}\sum_{u=1}^{N}e_u^2}
$$

Median error는 일반적인 샘플에서의 중앙 성능을 나타내고, P90 error는 전체 샘플 중 90%가 이 값 이하의 오차를 가진다는 의미이다. Max error는 최악의 실패 사례를 확인하기 위해 사용하였다. 실내 측위에서는 평균 성능뿐 아니라 worst-case 안정성도 중요하므로 다섯 지표를 함께 보고하였다.

평가 설계에서 가장 중요한 점은 validation set을 학습에 사용하지 않았다는 것이다. train set의 label은 residual target을 계산하는 데 사용되지만, validation set의 label은 학습 업데이트에 쓰이지 않고 최종 metric 계산에만 사용된다. 따라서 validation 성능은 제공 데이터 내부에서 확인할 수 있는 최소한의 일반화 지표이다. 물론 hidden test set 300개를 완전히 대체할 수는 없지만, 전체 700개를 모두 학습하고 같은 700개로 평가하는 방식보다는 훨씬 공정하다.

### 4.2 Baseline의 의미와 비교 공정성

본 보고서의 baseline은 AI를 전혀 사용하지 않은 inverse-distance weighted centroid이다. 중요한 점은 이 baseline이 GERT와 동일한 전처리 및 초기 위치 계산을 공유한다는 것이다. 즉, Baseline과 GERT의 차이는 “Transformer residual 보정기가 있는가”에 있다. 이 비교는 제안한 AI 보정기의 순수한 기여를 보기 위한 것이다.

단순 삼변측량이나 임의의 약한 방법을 baseline으로 쓰면 딥러닝 모델이 좋아 보이기 쉽다. 하지만 그런 비교는 공정하지 않다. 본 프로젝트에서는 실제 제안 알고리즘의 첫 단계인 weighted centroid를 baseline으로 사용하였다. 따라서 성능 향상은 MAD clipping 또는 초기 위치 계산 때문이 아니라, 동일한 초기 위치에 residual regression이 추가되었을 때 발생한 개선으로 해석할 수 있다.

Baseline 자체도 완전히 무의미한 방법은 아니다. 거리 역수 가중 중심은 각 기지국의 range measurement를 사용하여 사용자가 가까운 기지국 쪽에 위치할 것이라는 물리적 직관을 반영한다. 그러나 NLOS와 multipath로 인해 거리값이 왜곡되면 weighted centroid는 실제 위치에서 크게 벗어날 수 있다. GERT는 이 baseline의 한계를 residual feature와 Transformer 보정기로 보완한다.

### 4.3 Validation 결과와 오차 분포 해석

Validation set 기준 성능은 다음과 같다.

| 평가 지표 | Baseline | GERT | 절대 감소량 | 개선율 |
|---|---:|---:|---:|---:|
| Mean error | 23.86 m | 8.89 m | 14.97 m | 62.8% |
| Median error | 23.61 m | 7.65 m | 15.96 m | 67.6% |
| RMSE | 25.54 m | 10.45 m | 15.09 m | 59.1% |
| P90 error | 35.58 m | 15.65 m | 19.93 m | 56.0% |
| Max error | 43.84 m | 27.33 m | 16.51 m | 37.7% |

Mean error가 23.86 m에서 8.89 m로 감소했다는 것은, 평균적인 위치 추정 오차가 약 15 m 줄었다는 의미이다. 실내 측위 문제에서 15 m 수준의 오차 감소는 단순한 수치 개선이 아니라, 잘못된 방이나 구역으로 추정될 가능성을 크게 줄이는 수준의 개선이다.

Median error가 23.61 m에서 7.65 m로 감소한 점도 중요하다. Median은 일부 극단값보다 일반적인 샘플의 성능을 더 잘 보여준다. Mean과 Median이 모두 감소했기 때문에, 개선이 특정 몇 개 샘플에만 의존한 것이 아니라 전반적인 샘플 분포에서 나타났다고 볼 수 있다. 특히 Median의 개선율이 67.6%로 Mean 개선율보다 크다는 점은, 다수의 일반 샘플에서 residual 보정 효과가 안정적으로 나타났음을 의미한다.

RMSE는 25.54 m에서 10.45 m로 감소하였다. RMSE는 큰 오차를 제곱으로 반영하므로, 큰 실패 사례가 많으면 평균보다 더 크게 악화된다. RMSE가 Mean과 함께 크게 감소한 것은 GERT가 평균 성능뿐 아니라 큰 오차 샘플도 상당 부분 완화했음을 의미한다.

P90 error는 35.58 m에서 15.65 m로 감소하였다. P90 error는 전체 샘플 중 90%가 이 오차 이하에 들어온다는 의미이다. 즉, Baseline에서는 validation 사용자의 90%가 35.58 m 이하의 오차를 보였지만, GERT에서는 90%가 15.65 m 이하로 들어왔다. 이는 사용자 대부분에 대해 훨씬 안정적인 위치 추정이 가능해졌음을 나타낸다.

Max error는 43.84 m에서 27.33 m로 감소하였다. 다만 Max error의 개선율은 37.7%로 다른 지표보다 작다. 이는 극단적으로 어려운 일부 샘플에서는 초기 위치 자체가 크게 틀어지고, 그 결과 residual feature도 왜곡되어 Transformer가 완전히 보정하지 못할 수 있음을 보여준다. 따라서 Max error는 본 알고리즘의 남은 한계를 드러내는 지표이다.

### 4.4 학습 과정과 수렴 특성

학습 과정에서 train MSE는 전반적으로 감소하였다. validation MSE는 완전히 단조롭게 감소하지 않았는데, 이는 데이터 수가 제한되어 있고 validation set도 105개로 상대적으로 작기 때문이다. 따라서 마지막 epoch의 모델을 무조건 채택하는 것보다 validation loss 기준 best checkpoint를 저장하는 방식이 더 타당하다.

| Epoch | Train MSE | Validation MSE | 해석 |
|---:|---:|---:|---|
| 1 | 334.25 | 327.27 | 초기에는 보정량을 거의 학습하지 못한 상태이다. |
| 40 | 116.82 | 117.94 | residual 보정 패턴을 학습하면서 손실이 크게 감소한다. |
| 80 | 94.12 | 100.16 | 학습이 계속 진행되지만 감소 속도는 완만해진다. |
| 120 | 64.98 | 71.38 | validation 손실도 크게 줄어 일반화 개선이 나타난다. |
| 160 | 63.00 | 63.93 | train과 validation 손실이 비슷해 비교적 안정적이다. |
| 200 | 58.07 | 64.94 | train 손실은 더 낮지만 validation은 최저점과 약간 다를 수 있다. |

이 결과는 200 epoch가 “데이터 200개만 학습한다”는 뜻이 아니라, train set 595개 전체를 200회 반복하여 학습한다는 뜻이다. 모델은 각 epoch에서 여러 mini-batch를 통해 residual 보정 벡터를 학습한다. epoch 1에서 validation MSE가 327.27인 것은 초기 모델이 거의 랜덤한 보정량을 출력한다는 의미이다. epoch 40에서 validation MSE가 117.94까지 급격히 감소한 것은 모델이 초기 위치 오차의 주요 방향성을 빠르게 학습했음을 의미한다. epoch 120 이후에는 감소 속도가 완만해지며, 이는 residual regression 문제에서 학습 가능한 주요 패턴을 상당 부분 학습한 뒤 세부 샘플 차이에 맞추는 단계로 들어갔기 때문으로 해석된다.

Train MSE와 validation MSE의 차이가 지나치게 벌어지지 않았다는 점도 중요하다. 작은 데이터셋에서 과도한 모델을 사용하면 train loss만 계속 감소하고 validation loss는 증가하는 전형적인 과적합이 나타난다. 본 모델은 dropout, weight decay, compact architecture를 사용했기 때문에 validation loss가 함께 감소하는 구간을 유지하였다. 이는 모델 크기와 정규화 설정이 데이터 규모에 비해 과도하지 않았음을 뒷받침한다.

### 4.5 파라미터 민감도 실험 해석

30 epoch 보조 실험은 최종 200 epoch 모델의 절대 성능을 대체하는 실험은 아니지만, 설계 선택의 타당성을 보여준다.

| 실험 | Mean | RMSE | P90 | Max | 핵심 해석 |
|---|---:|---:|---:|---:|---|
| Baseline | 23.86 | 25.54 | 35.58 | 43.84 | AI 보정 없이 초기 위치만 사용하면 오차가 크다. |
| Full GERT 30ep | 12.92 | 14.70 | 22.94 | 27.89 | 짧은 학습만으로도 baseline을 크게 개선한다. |
| Distance-only 30ep | 15.66 | 17.27 | 25.28 | 31.37 | residual을 제거하면 모든 주요 지표가 악화된다. |
| Small model 30ep | 16.36 | 18.16 | 26.40 | 36.82 | 모델 표현력이 부족하면 residual을 충분히 활용하지 못한다. |
| Large model 30ep | 13.34 | 15.06 | 22.74 | 32.28 | 모델을 키워도 성능이 자동으로 좋아지지 않는다. |
| Low LR 30ep | 12.91 | 14.85 | 22.29 | 30.72 | 낮은 learning rate는 안정적이나 최종 설정 대비 큰 이점은 없다. |

Distance-only 실험은 residual feature의 중요성을 직접 보여준다. 입력에서 residual channel을 제거하면 모델 구조와 파라미터 수는 동일하지만 Mean error가 12.92 m에서 15.66 m로 증가한다. 이는 모델이 단순 거리값만 보는 것보다, 거리 방정식의 불일치를 함께 볼 때 더 좋은 보정 방향을 찾는다는 뜻이다.

Small model 실험은 모델 용량의 하한을 보여준다. d_model을 16으로 줄이면 파라미터 수는 3,922개로 감소하지만 Mean error는 16.36 m까지 악화된다. 이는 residual feature가 아무리 유용해도, 그 패턴을 표현할 최소한의 비선형 모델 용량이 필요함을 의미한다.

Large model 실험은 모델 용량의 상한에 대한 시사점을 준다. d_model을 64로 늘리면 파라미터 수가 27,634개로 증가하지만, 30 epoch 기준 성능은 Full GERT보다 좋아지지 않았다. 이는 제공 데이터 700개 조건에서 모델을 키우는 것보다 물리적으로 의미 있는 feature를 설계하는 것이 더 중요하다는 판단을 뒷받침한다.

Low LR 실험은 학습률 선택의 민감도를 확인하기 위한 것이다. learning rate 0.001도 baseline보다 크게 좋지만, 30 epoch 기준 기본 learning rate 0.005와 큰 차이를 보이지 않았다. 최종 모델에서는 200 epoch 동안 빠르게 수렴하면서 발산하지 않는 0.005를 사용하였다.

### 4.6 최종 제출 모델 sanity check

최종 제출용 model.pt는 검증 실험 이후 전체 700개 샘플로 재학습하여 생성하였다. 아래 결과는 같은 700개 학습 데이터에서 다시 평가한 값이므로 hidden test 성능이나 일반화 성능으로 해석하지 않는다. 이 표는 최종 저장된 모델이 학습 데이터에 대해 residual 보정을 정상적으로 수행하는지 확인하기 위한 sanity check이다.

| 평가 지표 | All-data Baseline | All-data GERT final | 개선율 |
|---|---:|---:|---:|
| Mean error | 23.35 m | 6.84 m | 70.7% |
| Median error | 22.74 m | 5.84 m | 74.3% |
| RMSE | 25.82 m | 8.22 m | 68.2% |
| P90 error | 38.18 m | 12.25 m | 67.9% |
| Max error | 56.38 m | 32.86 m | 41.7% |

이 sanity check는 제출용 model.pt가 단순히 baseline을 반환하는 것이 아니라 실제로 residual 보정을 수행하고 있음을 보여준다. 다만 이 값은 같은 700개 데이터에서 계산되었으므로 validation 결과보다 더 좋게 나오는 것이 자연스럽다. 따라서 본 보고서의 주요 성능 주장은 validation 결과를 기준으로 한다.

전체 데이터 재학습을 수행한 이유는 hidden test 제출 모델을 만들 때 사용 가능한 정보를 최대한 활용하기 위해서이다. validation 성능 평가는 595개 학습, 105개 검증으로 공정하게 수행하고, 최종 모델은 제출 직전에 전체 제공 데이터를 사용하여 학습한다. 이는 머신러닝 실무에서도 흔히 쓰는 절차이다. 모델 선택과 성능 주장은 validation으로 하고, 최종 배포 모델은 사용 가능한 모든 학습 데이터를 활용한다.

### 4.7 사고와 구현의 적합성

본 프로젝트의 핵심 사고는 데이터가 부족한 환경에서 인공지능에게 위치 전체를 맡기지 않는 것이다. 제공 데이터가 제한적이므로, 좌표를 직접 예측하는 큰 모델은 학습 데이터의 공간 분포를 외우는 방향으로 과적합될 위험이 크다. 따라서 본 알고리즘은 물리 기반 초기 위치를 먼저 만들고, 모델은 그 초기 위치에서 실제 위치까지의 보정량만 학습하도록 제한하였다.

이 구현은 WiFi RTT 측위 문제의 특성과도 맞다. 측정 거리 자체는 물리적 의미를 갖고, 초기 위치는 부정확하더라도 대략적인 기하학 정보를 제공한다. 여기에 residual feature를 추가하면, 모델은 단순히 거리 크기만 보는 것이 아니라 거리 방정식의 모순을 관찰할 수 있다. 따라서 제한된 데이터에서도 baseline 대비 유의미한 오차 감소가 나타난 것으로 판단한다.

또한 Compact Transformer를 사용한 이유도 문제 구조와 맞다. 18개 기지국은 서로 독립적인 숫자 18개가 아니라 동일한 사용자 위치를 서로 다른 방향에서 관측한 센서 집합이다. 특정 기지국의 거리값이 신뢰 가능한지는 다른 기지국들의 거리 및 residual 패턴과 함께 판단되어야 한다. Transformer는 token 사이의 관계를 학습할 수 있으므로, 18개 앵커를 token으로 해석하는 설계가 자연스럽다.

다만 본 프로젝트는 모든 가능한 최적화를 수행한 것은 아니다. WLS 초기화, 기지국별 reliability modeling, attention weight 분석, 여러 seed 반복 평가 등은 수행하지 않았다. 따라서 본 보고서의 주장은 “GERT가 모든 측위 알고리즘보다 우수하다”가 아니다. 보다 정확한 주장은 “제공 데이터와 자체 validation 조건에서, 물리 기반 weighted centroid baseline에 compact Transformer residual correction을 추가하면 오차 분포가 크게 개선된다”이다. 이와 같이 주장 범위를 제한하는 것이 보고서의 신뢰성을 높인다고 판단하였다.

### 4.8 hidden test 관점의 일반화 논의

과제의 실제 성능은 조교가 보유한 hidden test set에서 결정된다. hidden test에는 제공 데이터와 동일한 형식의 \(d_hat\), \(BS_positions\), \(p\)가 들어 있지만, 사용자는 그 좌표와 거리 패턴을 학습 중에 볼 수 없다. 따라서 hidden test에서 중요한 것은 학습 데이터 좌표를 암기하는 능력이 아니라, 새로운 거리 패턴에 대해 안정적으로 위치를 추정하는 능력이다.

GERT는 hidden test 일반화를 위해 다음 설계를 사용하였다.

| 설계 | hidden test에서 기대되는 장점 |
|---|---|
| 사용자 수 동적 처리 | hidden test 사용자 수가 300명이어도 동일하게 동작한다. |
| shape 정규화 | \((2,18)\), \((18,2)\), \((18,N)\) 형태 차이에 안정적으로 대응한다. |
| MAD clipping | hidden test에서 극단적 거리값이 있어도 입력 분포가 완전히 무너지지 않는다. |
| residual regression | 좌표 직접 암기보다 거리 방정식 모순 패턴에 의존한다. |
| compact model | 작은 데이터에서 과적합 위험을 줄인다. |
| batch inference | hidden test 전체 사용자에 대해 빠르게 추론한다. |

물론 hidden test 분포가 제공 데이터와 크게 다르면 성능은 변할 수 있다. 예를 들어 hidden 사용자 위치가 제공 데이터가 거의 포함하지 않는 공간에 집중되어 있거나, 특정 기지국의 NLOS bias가 학습 데이터와 다르게 나타나면 모델의 보정량이 부정확해질 수 있다. 그러나 초기 위치와 residual feature는 데이터 분포가 바뀌어도 물리적으로 의미를 유지하므로, 좌표 직접 회귀보다 더 안정적일 것으로 기대한다.

### 4.9 장점, 한계, 향후 개선

본 알고리즘의 장점은 다음과 같다.

| 장점 | 설명 |
|---|---|
| 물리 기반 구조 | 거리값을 바로 좌표로 매핑하지 않고, range-based 초기 위치를 먼저 계산한다. |
| residual regression | 모델이 절대 좌표가 아니라 초기 위치의 오차만 보정하므로 과적합 위험을 줄인다. |
| 앵커 token화 | 18개 기지국을 token으로 보고 앵커 간 상관관계를 학습할 수 있다. |
| 경량 구조 | 9,778개 파라미터로 제한하여 데이터 수 대비 과도한 모델을 피했다. |
| 실행 안정성 | main.py는 전체 사용자를 batch로 처리하고, model.pt를 불러와 빠르게 추론한다. |
| 보고서와 코드 일치성 | train.py와 main.py의 전처리 및 모델 구조를 맞추었다. |

한계도 존재한다.

| 한계 | 설명 |
|---|---|
| 초기 위치 의존성 | weighted centroid가 크게 벗어나면 residual 자체가 왜곡될 수 있다. |
| validation 규모 | validation set이 105개뿐이므로 split에 따라 metric 변동이 생길 수 있다. |
| attention 해석 미수행 | attention weight를 별도로 시각화하지 않았으므로, 어떤 기지국을 신뢰했는지 정량적으로 증명하지는 못했다. |
| WLS 비교 부족 | WLS trilateration, robust least squares, Kalman filter와의 직접 비교는 수행하지 않았다. |
| hidden test 불확실성 | 제공 데이터 700개와 hidden 300개의 분포가 다르면 성능이 변할 수 있다. |

향후 개선 방향은 다음과 같다. 첫째, weighted centroid 대신 WLS 기반 초기 위치를 사용하면 초기 위치의 물리적 정확도를 높일 수 있다. 둘째, residual channel 제거, distance-only Transformer, MLP residual model을 더 긴 epoch와 여러 seed에서 비교하면 feature의 기여를 더 명확히 검증할 수 있다. 셋째, attention weight 또는 gradient-based attribution을 분석하면 어떤 기지국 token이 보정에 큰 영향을 주었는지 설명 가능성을 높일 수 있다. 넷째, 이동 궤적 데이터가 주어진다면 Kalman filter 또는 temporal Transformer와 결합하여 시간적 연속성을 반영할 수 있다. 다섯째, 기지국별 신뢰도나 NLOS likelihood를 별도 head로 예측하는 multi-task learning 구조를 도입할 수도 있다.

### 4.10 최종 디스커션

본 프로젝트의 성능 개선은 단순히 Transformer를 사용했기 때문에 발생한 것이 아니다. 핵심은 Transformer 이전에 어떤 물리 정보를 모델에 제공했는가이다. 원시 거리값만으로는 모델이 기하학적 일관성을 스스로 학습해야 한다. 반면 본 프로젝트는 초기 위치와 residual을 통해 거리 방정식의 불일치를 명시적으로 제공하였다. 이 때문에 작은 모델과 제한된 데이터에서도 의미 있는 성능 개선이 가능했다.

또한 본 프로젝트는 성능과 참신성 사이의 균형을 고려하였다. 단순 WLS나 robust least-squares는 통신 측위 문제에 매우 적합하지만, 과제의 참신성 측면에서는 기존 방법의 조합으로 보일 수 있다. 반대로 매우 큰 딥러닝 모델이나 강화학습 접근은 참신할 수 있으나, 제한된 데이터와 hidden test 조건에서 성능과 안정성이 불안할 수 있다. GERT는 이 사이에서 물리 기반 해석성과 AI 기반 보정의 장점을 결합하려는 설계이다.

최종적으로 본 프로젝트의 결론은 다음과 같다. WiFi RTT 실내 측위에서 원시 거리값은 NLOS와 multipath 때문에 불완전하지만, 여전히 물리적 의미를 가진다. 따라서 거리값을 완전히 black-box 모델에 맡기기보다, 먼저 range-based 초기 위치를 만들고, 그 위치에서 발생하는 기하학적 residual을 학습 입력으로 사용하는 것이 제한된 데이터 환경에서 합리적이다. Validation 결과에서 GERT는 weighted centroid baseline 대비 Mean error, Median error, RMSE, P90 error, Max error를 모두 개선하였다. 이는 제안한 residual correction 구조가 단순 초기 위치 추정보다 더 안정적인 위치 추정을 수행함을 보여준다.

## 5. Reference

추가된 평가 조건을 반영하여, 단순히 참고문헌 목록만 제시하지 않고 각 reference에서 무엇을 참고했는지와 본 프로젝트에서 직접 설계한 부분이 무엇인지 구분하였다. 아래 표의 “참고한 내용”은 해당 문헌에서 본 프로젝트의 문제 이해 또는 설계 방향에 영향을 준 부분이고, “본 프로젝트에서 직접 수행한 부분”은 기존 연구를 그대로 구현하지 않고 과제 조건에 맞게 새로 구성한 부분이다.

| Reference | 참고한 내용 | 본 프로젝트에서 직접 수행한 부분 및 차별점 |
|---|---|---|
| [1] Liu et al., 2007 | 무선 실내 측위 기술이 range-based, fingerprinting, proximity 등 다양한 방식으로 나뉘며, 실내 환경에서 multipath와 NLOS가 주요 오차 요인이라는 큰 문제의식을 참고하였다. 또한 WiFi 계열 실내 측위가 환경 의존적이고, 단순 거리 기반 방법만으로는 안정적인 성능을 얻기 어렵다는 점을 motivation에 반영하였다. | 본 프로젝트는 survey에서 정리된 여러 기술을 단순히 나열하거나 구현하지 않았다. WiFi RTT 데이터의 18개 거리값을 사용하여 inverse-distance weighted centroid로 초기 위치를 만들고, 그 초기 위치와 거리 방정식 사이의 residual을 Transformer 입력 feature로 직접 설계하였다. 즉, 본 프로젝트의 핵심은 survey의 일반론을 바탕으로 제한된 데이터셋에 맞는 hybrid residual correction 파이프라인을 직접 구성한 것이다. |
| [2] Gezici et al., 2005 | UWB localization에서 range measurement, NLOS, multipath, 거리 측정 오차가 위치추정 정확도에 큰 영향을 준다는 점을 참고하였다. 특히 거리 기반 측위에서는 단순 좌표 회귀보다 거리 오차의 물리적 의미를 고려해야 한다는 문제의식을 얻었다. | 해당 논문은 UWB 센서 네트워크와 positioning aspect를 다루지만, 본 프로젝트는 WiFi RTT 기반 데이터셋을 사용하였다. 또한 본 프로젝트는 WLS 또는 확률적 bounds를 직접 구현하지 않고, 초기 위치에서 발생하는 기하학적 residual을 학습 feature로 사용하였다. 따라서 “거리 측정 오차를 물리적으로 해석해야 한다”는 방향은 참고했지만, 최종 알고리즘은 compact Transformer residual regression으로 새로 설계하였다. |
| [3] Rousseeuw and Croux, 1993 | median 기반 robust scale estimator의 필요성과, 평균·표준편차가 outlier에 민감하다는 통계적 문제의식을 참고하였다. 본 프로젝트의 MAD 기반 clipping은 거리값의 극단 오차가 전체 위치 추정을 무너뜨리지 않도록 하기 위한 전처리 근거로 사용되었다. | 해당 논문은 MAD의 대안적 robust scale estimator를 제안하는 통계 논문이며, 실내 측위 알고리즘을 제안한 논문은 아니다. 본 프로젝트에서는 그 robust statistics 관점을 WiFi RTT 거리 행렬의 기지국별 전처리에 적용하였다. 또한 이상치를 삭제하지 않고 정상 범위로 clipping하여 Transformer 입력 token 수를 유지하는 방식은 본 과제의 입력 shape 조건에 맞게 직접 설계한 부분이다. |
| [4] Vaswani et al., 2017 | Transformer의 self-attention이 token 간 관계를 학습할 수 있다는 핵심 구조를 참고하였다. 본 프로젝트에서는 문장의 단어 대신 18개 기지국을 token으로 해석할 수 있다고 보았다. | 해당 논문은 자연어 처리용 sequence-to-sequence Transformer를 제안하였다. 본 프로젝트는 이를 그대로 사용하지 않고, 18개 앵커 token과 2개 feature 거리·residual을 입력으로 하는 1-layer Compact Transformer Encoder로 축소하였다. 대규모 NLP 모델이 아니라, 700개 데이터와 10분 실행 제한에 맞는 9,778개 파라미터의 경량 residual regressor를 직접 구성한 점이 차별점이다. |
| [5] Raissi et al., 2019 | Physics-informed learning의 핵심인 “물리 법칙과 신경망을 결합하면 데이터가 적은 문제에서 학습을 안정화할 수 있다”는 관점을 참고하였다. 물리 정보를 학습 모델에 넣는 것이 단순 black-box 딥러닝보다 유리할 수 있다는 방향성을 얻었다. | 해당 논문은 PDE residual을 loss function에 포함하는 PINN 구조를 제안한다. 본 프로젝트는 PDE를 풀지 않고, WiFi RTT 거리 방정식의 residual을 입력 feature로 직접 주입하였다. 즉, physics-informed loss가 아니라 physics-informed feature engineering을 선택하였다. 이는 계산을 단순하게 유지하고, main.py 추론 시간을 줄이며, hidden test 환경에서 안정적으로 실행되도록 하기 위한 본 프로젝트의 직접 설계이다. |

각 reference가 본 프로젝트에 미친 영향을 더 구체적으로 정리하면 다음과 같다.

| 구분 | 기존 연구의 역할 | 본 프로젝트에서의 재해석 |
|---|---|---|
| 실내 측위 문제 정의 | [1]은 실내 측위에서 환경 의존성과 NLOS 문제가 중요함을 정리한다. | WiFi RTT 데이터에서 단순 거리 기반 추정이 불안정하다는 motivation으로 사용하였다. |
| 거리 기반 측위와 NLOS | [2]는 range measurement 기반 위치추정에서 거리 오차가 핵심임을 설명한다. | 거리값을 직접 버리지 않고 초기 위치와 residual을 만드는 물리 기반 feature로 사용하였다. |
| robust statistics | [3]은 outlier에 강한 scale 추정의 필요성을 보여준다. | 기지국별 MAD clipping으로 극단 RTT 값을 완화하되, 입력 token 수는 유지하였다. |
| self-attention | [4]는 token 간 관계를 학습하는 구조를 제안한다. | 18개 앵커를 token으로 보고 거리와 residual의 상호 패턴을 학습하도록 변형하였다. |
| physics-informed learning | [5]는 물리 법칙과 신경망 결합의 유효성을 제시한다. | 물리 방정식을 loss에 넣는 대신, 거리 방정식 residual을 입력 feature로 직접 주입하였다. |

Reference 목록은 다음과 같다.

[1] H. Liu, H. Darabi, P. Banerjee, and J. Liu, “Survey of Wireless Indoor Positioning Techniques and Systems,” IEEE Transactions on Systems, Man, and Cybernetics, Part C: Applications and Reviews, vol. 37, no. 6, pp. 1067–1080, 2007, doi: 10.1109/TSMCC.2007.905750.

[2] S. Gezici, Z. Tian, G. B. Giannakis, H. Kobayashi, A. F. Molisch, H. V. Poor, and Z. Sahinoglu, “Localization via Ultra-Wideband Radios: A Look at Positioning Aspects for Future Sensor Networks,” IEEE Signal Processing Magazine, vol. 22, no. 4, pp. 70–84, 2005, doi: 10.1109/MSP.2005.1458289.

[3] P. J. Rousseeuw and C. Croux, “Alternatives to the Median Absolute Deviation,” Journal of the American Statistical Association, vol. 88, no. 424, pp. 1273–1283, 1993, doi: 10.1080/01621459.1993.10476408.

[4] A. Vaswani et al., “Attention Is All You Need,” Advances in Neural Information Processing Systems, 2017.

[5] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” Journal of Computational Physics, vol. 378, pp. 686–707, 2019, doi: 10.1016/j.jcp.2018.10.045.
