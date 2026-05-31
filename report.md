# Geo-Explainable Residual Transformer for Robust WiFi RTT Indoor Localization

## 1. 모티베이션 & 인트로

WiFi RTT 기반 실내 측위는 사용자와 여러 기지국 사이의 거리 추정값을 이용하여 사용자 위치를 계산하는 range-based localization 문제이다. 이상적인 환경에서는 여러 기지국까지의 거리 원이 한 점 근처에서 만나는 삼변측량 구조를 기대할 수 있다. 그러나 실제 실내 환경에서는 벽, 장애물, 반사체, 사람의 이동 등으로 인해 NLOS와 multipath 오차가 발생한다. 이때 일부 기지국의 거리 추정값은 실제 거리보다 크게 왜곡되고, 단순 삼변측량이나 단순 평균 기반 위치 추정은 큰 오차를 갖게 된다.

중간발표 전후의 실험에서는 MAD 기반 이상치 처리, DBSCAN 클러스터링, MCC 기반 강건 추정 등 통계적 방법을 우선 검토하였다. 이러한 방법은 튀는 거리값을 일부 완화할 수 있었지만, 18개 기지국의 상대적 배치와 특정 기지국 조합에서 나타나는 비선형 오차 패턴을 충분히 반영하기 어려웠다. 특히 모든 기지국에 같은 통계 기준을 적용하면, 어떤 기지국이 실제로 위치 추정에 유리한지 또는 어떤 기지국이 NLOS 영향을 강하게 받았는지까지 판단하기 어렵다.

이 한계에서 출발하여 본 프로젝트에서는 순수 통계 필터 또는 순수 딥러닝 대신, 기하학 기반 초기 위치 추정과 머신러닝 기반 residual 보정을 결합한 GERT, Geo-Explainable Residual Transformer를 설계하였다. 핵심 아이디어는 인공지능이 사용자 좌표를 처음부터 직접 예측하게 하지 않는 것이다. 먼저 거리값으로 물리적으로 그럴듯한 초기 위치를 만든 뒤, 그 초기 위치가 실제 위치로 이동하기 위해 필요한 보정량만 학습하게 한다. 이렇게 하면 모델이 2차원 좌표 공간 전체를 외우는 부담이 줄어들고, 제한된 데이터 환경에서도 일반화 가능성이 높아진다.

본 프로젝트에서 사용한 데이터와 학습 설정은 다음과 같다.

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

전체 알고리즘은 먼저 MAD 기반 clipping으로 극단적인 거리값을 완화하고, inverse-distance weighted centroid로 초기 위치를 계산한다. 이후 각 기지국에서 측정된 거리와 초기 위치 기준 물리적 거리의 차이를 residual feature로 만들고, 18개 기지국을 Transformer의 token으로 취급하여 보정 벡터를 예측한다. 최종 위치는 초기 위치에 이 보정 벡터를 더해 계산된다.

## 2. 알고리즘 설명

### 2.1 MAD 기반 거리값 정제

실내 RTT 거리값은 NLOS와 multipath 때문에 일부 기지국에서 비정상적으로 큰 값을 가질 수 있다. 이를 완화하기 위해 각 기지국별 거리 분포에 대해 median과 MAD를 계산하고, Modified Z-score가 기준값을 초과하는 값을 정상 범위의 최솟값과 최댓값 사이로 clipping하였다. 본 프로젝트에서 사용한 기준값은 다음과 같다.

| 항목 | 값 |
|---|---:|
| Modified Z-score threshold | 3.5 |
| clipping 기준 | 기지국별 정상 거리 범위 |
| 목적 | 극단적 거리값 완화 및 입력 shape 유지 |

MAD와 Modified Z-score는 다음과 같이 정의된다.

$$
\text{MAD}_i = \text{median}\left(|d_i - \text{median}(d_i)|\right)
$$

$$
z_i = 0.6745 \cdot \frac{d_i - \text{median}(d_i)}{\text{MAD}_i}
$$

여기서 i는 기지국 인덱스를 의미한다. 이상치를 완전히 삭제하지 않고 clipping한 이유는 모든 사용자에 대해 18개 기지국 입력을 유지해야 Transformer 입력 구조가 안정적으로 유지되기 때문이다.

### 2.2 거리 역수 가중 중심 기반 초기 위치

본 알고리즘은 엄밀한 비선형 least-squares 삼변측량을 직접 푸는 방식은 아니다. 대신 삼변측량의 핵심인 range measurement 정보를 활용하되, 계산 안정성과 속도를 위해 inverse-distance weighted centroid를 초기 위치 추정기로 사용한다. 측정 거리가 짧은 기지국일수록 사용자와 가까울 가능성이 높다고 보고 더 큰 가중치를 부여한다.

$$
\tilde{w}_i = \frac{1}{\hat{d}_i + \epsilon}
$$

$$
w_i = \frac{\tilde{w}_i}{\sum_{j=1}^{18}\tilde{w}_j}
$$

$$
\hat{p}_{init} = \sum_{i=1}^{18}w_i p_{bs,i}
$$

이 방식은 모든 사용자에 대해 행렬 연산으로 빠르게 계산할 수 있고, 딥러닝 모델에 물리적으로 그럴듯한 출발점을 제공한다. 따라서 본 알고리즘은 “range-based localization 원리에 기반한 초기 추정 후 residual correction” 구조로 해석하는 것이 가장 정확하다.

### 2.3 기하학적 residual feature

초기 위치가 계산되면 각 기지국에 대해 측정 거리와 초기 위치 기준 물리적 거리의 차이를 계산한다. 이 값이 본 프로젝트의 핵심 feature인 geometric residual이다.

$$
r_i = \hat{d}_i - \left\|\hat{p}_{init} - p_{bs,i}\right\|_2
$$

이 residual은 초기 위치가 실제 거리 방정식과 얼마나 모순되는지 나타낸다. NLOS 영향을 받은 기지국은 실제 거리보다 긴 측정값을 만들 가능성이 크므로, 해당 기지국의 residual 패턴이 다른 기지국과 다르게 나타날 수 있다. 따라서 모델 입력은 단순한 거리 18개가 아니라, 각 기지국별 거리와 residual의 쌍으로 구성된다.

| 입력 구성 | Shape | 의미 |
|---|---:|---|
| 거리 feature | 18 × 1 | 기지국별 RTT 기반 거리 추정값 |
| residual feature | 18 × 1 | 측정 거리와 초기 위치 기준 물리 거리의 차이 |
| Transformer 입력 | 18 × 2 | 기지국 18개를 token으로 하는 sequence |

### 2.4 Compact Residual Transformer

18개 기지국을 각각 하나의 token으로 보고, 각 token에 거리와 residual 두 feature를 부여한다. Transformer Encoder는 이 token sequence를 받아 기지국 간 상관관계를 학습하고, 최종적으로 초기 위치에 더할 2차원 보정 벡터를 출력한다.

$$
\Delta \hat{p} = f_\theta(X)
$$

$$
\hat{p}_{final} = \hat{p}_{init} + \Delta \hat{p}
$$

모델이 직접 예측하는 값은 절대 좌표가 아니라 보정량이다. 학습 label도 실제 위치와 초기 위치의 차이로 정의하였다.

$$
y = p_{true} - \hat{p}_{init}
$$

$$
\mathcal{L}(\theta)=\frac{1}{N}\sum_{k=1}^{N}\left\|f_\theta(X_k)-(p_{true,k}-\hat{p}_{init,k})\right\|_2^2
$$

모델 구조는 데이터 수가 제한된 점을 고려하여 작게 설계하였다.

| 구성 요소 | 설정 |
|---|---:|
| 입력 token 수 | 18 |
| token feature 수 | 2 |
| embedding dimension | 32 |
| attention head 수 | 4 |
| Transformer Encoder layer 수 | 1 |
| feedforward dimension | 64 |
| encoder dropout | 0.2 |
| output head dropout | 0.1 |
| 학습 파라미터 수 | 9,778 |

### 2.5 관련 연구와의 차별점

WiFi, UWB, RTT 기반 실내 측위 연구들은 일반적으로 거리 기반 삼변측량, probabilistic localization, fingerprinting, robust estimation을 사용한다. 본 프로젝트는 이 중 range-based localization의 해석 가능성을 유지하면서도, residual을 학습 feature로 주입한다는 점에서 차이가 있다. Liu 등의 indoor positioning survey는 무선 실내 측위 기술 전반을 정리하지만, 본 프로젝트처럼 제한된 데이터에서 기하학적 residual을 Transformer 입력으로 구성하는 구조를 직접 제안하지는 않는다.

UWB localization 연구에서는 거리 측정 오차와 NLOS 문제가 중요한 요소로 다뤄진다. 본 프로젝트도 같은 문제의식을 공유하지만, WLS나 확률 필터만으로 오차를 줄이는 대신 초기 위치의 기하학적 모순을 머신러닝 입력으로 사용한다.

Transformer 연구는 원래 sequence token 사이의 관계를 학습하기 위한 구조를 제안하였다. 본 프로젝트에서는 문장 token 대신 18개 기지국을 token으로 간주한다. 단, 대규모 Transformer를 사용하지 않고 1-layer compact encoder만 사용하여 700개 데이터 환경에 맞게 축소하였다.

Physics-informed neural network 계열 연구는 보통 물리 방정식을 손실 함수에 추가한다. 본 프로젝트는 물리 방정식을 손실에 직접 넣기보다, 측정 거리와 초기 위치 기반 거리의 차이를 residual feature로 입력에 직접 주입한다. 이 방식은 구현이 간단하고, 제출 환경의 실행 시간 제한 안에서 안정적으로 동작한다.

## 3. Agent AI 활용 방안

본 프로젝트에서는 Agent AI를 알고리즘 설계의 대체자가 아니라 구현 보조 및 검토 도구로 사용하였다. 핵심 알고리즘의 방향, 즉 “거리 기반 초기 위치를 먼저 계산하고, 딥러닝은 위치 자체가 아니라 residual 보정량만 학습하게 한다”는 판단은 WiFi RTT 데이터의 물리적 특성과 과적합 위험을 고려하여 직접 결정하였다.

본인이 주도한 부분은 다음과 같다.

| 구분 | 본인이 수행한 역할 |
|---|---|
| 문제 정의 | WiFi RTT 실내 측위에서 NLOS와 multipath가 핵심 오차 요인임을 분석 |
| 알고리즘 구조 | 좌표 직접 예측 대신 residual regression 구조 선택 |
| feature 설계 | 거리와 기하학적 residual을 함께 사용하는 입력 설계 |
| 모델 선택 | 18개 기지국을 token으로 보는 Compact Transformer 구조 선택 |
| 평가 해석 | validation 성능과 all-data sanity check를 구분하여 해석 |
| 제출 구성 | main.py, train.py, model.pt, report.md의 역할 정리 |

Agent AI는 다음 업무에 보조적으로 활용하였다.

| 구분 | Agent AI 활용 내용 |
|---|---|
| 코드 구조화 | PyTorch 기반 Compact Residual Transformer 구현 보조 |
| 디버깅 | d_hat, BS_positions, p의 shape 불일치 문제 점검 |
| 학습 코드 점검 | train.py와 main.py의 전처리 일치성 확인 |
| 모델 저장 | model.pt 저장 및 로딩 흐름 검토 |
| 결과 정리 | metric 계산 결과를 markdown 표로 정리 |
| 보고서 점검 | 과장된 표현 제거 및 구현과 설명의 일치성 검토 |

Agent AI가 제안한 내용은 그대로 채택하지 않고, 실제 코드 실행 결과와 제출 규격에 맞는지 확인한 뒤 반영하였다. 특히 report.md와 train.py가 참신성 및 일치성 평가의 중심이라는 점을 고려하여, 보고서 설명이 실제 학습 코드와 어긋나지 않도록 수정하였다.

## 4. 결과 도출 & 디스커션

### 4.1 자체 평가 방식

전체 700개 제공 데이터를 무작위로 섞은 뒤, seed 42 기준으로 train set과 validation set을 분리하였다. 모델은 train set만 사용해 학습하고, validation set은 성능 평가에만 사용하였다. 최종 제출용 model.pt는 검증 실험 이후 전체 700개 샘플을 이용해 다시 학습하였지만, 전체 데이터로 재학습한 뒤 같은 데이터에서 계산한 성능은 일반화 성능으로 주장하지 않았다.

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

여기서 epochs는 학습 샘플 수를 뜻하지 않는다. epochs 200은 train set 전체를 200회 반복하여 학습했다는 의미이다. 과적합을 줄이기 위해 validation loss가 가장 낮은 checkpoint를 기준으로 검증 성능을 평가하였다.

### 4.2 Validation 결과

Baseline은 AI 보정 전의 inverse-distance weighted centroid이다. GERT는 같은 MAD clipping과 같은 초기 위치를 사용하되, 추가로 distance와 residual feature를 Transformer에 입력하여 보정 벡터를 예측한다. 따라서 비교는 “기하학적 초기 추정만 사용한 경우”와 “동일한 초기 추정에 머신러닝 residual 보정을 추가한 경우”의 비교이다.

| 평가 지표 | Baseline | GERT | 개선율 |
|---|---:|---:|---:|
| Mean error | 23.86 m | 8.89 m | 62.8% |
| Median error | 23.61 m | 7.65 m | 67.6% |
| RMSE | 25.54 m | 10.45 m | 59.1% |
| P90 error | 35.58 m | 15.65 m | 56.0% |
| Max error | 43.84 m | 27.33 m | 37.7% |

Mean, Median, RMSE, P90 error가 모두 감소했으므로, 특정 샘플에서만 우연히 좋아진 결과라기보다 오차 분포 전체가 낮아진 것으로 해석할 수 있다. Max error의 개선율은 평균 오차나 P90보다 작다. 이는 극단적으로 왜곡된 일부 위치에서는 초기 위치 자체가 크게 틀어져 residual feature도 불안정해질 수 있음을 의미한다.

### 4.3 학습 과정 해석

학습 과정에서 train MSE는 전반적으로 감소하였다. validation MSE는 완전히 단조롭게 감소하지 않았는데, 이는 데이터 수가 제한되어 있고 validation set도 상대적으로 작기 때문이다. 이 때문에 마지막 epoch의 모델을 무조건 채택하지 않고, validation loss 기준 best checkpoint를 저장하는 방식이 더 타당하다.

| Epoch | Train MSE | Validation MSE |
|---:|---:|---:|
| 1 | 334.25 | 327.27 |
| 40 | 116.82 | 117.94 |
| 80 | 94.12 | 100.16 |
| 120 | 64.98 | 71.38 |
| 160 | 63.00 | 63.93 |
| 200 | 58.07 | 64.94 |

### 4.4 최종 제출 모델 sanity check

최종 제출용 model.pt는 검증 실험 이후 전체 700개 샘플로 재학습하여 생성하였다. 아래 결과는 같은 700개 학습 데이터에서 다시 평가한 값이므로 hidden test 성능이나 일반화 성능으로 해석하지 않는다. 이 표는 최종 저장된 모델이 학습 데이터에 대해 residual 보정을 정상적으로 수행하는지 확인하기 위한 sanity check이다.

| 평가 지표 | All-data Baseline | All-data GERT final | 개선율 |
|---|---:|---:|---:|
| Mean error | 23.35 m | 7.06 m | 69.8% |
| Median error | 22.74 m | 6.16 m | 72.9% |
| RMSE | 25.82 m | 8.55 m | 66.9% |
| P90 error | 38.18 m | 12.42 m | 67.5% |
| Max error | 56.38 m | 33.86 m | 39.9% |

### 4.5 사고와 구현의 적합성

본 프로젝트의 핵심 사고는 데이터가 부족한 환경에서 인공지능에게 위치 전체를 맡기지 않는 것이다. 제공 데이터가 제한적이므로, 좌표를 직접 예측하는 큰 모델은 학습 데이터의 공간 분포를 외우는 방향으로 과적합될 위험이 크다. 따라서 본 알고리즘은 물리 기반 초기 위치를 먼저 만들고, 모델은 그 초기 위치에서 실제 위치까지의 보정량만 학습하도록 제한하였다.

이 구현은 WiFi RTT 측위 문제의 특성과도 맞다. 측정 거리 자체는 물리적 의미를 갖고, 초기 위치는 부정확하더라도 대략적인 기하학 정보를 제공한다. 여기에 residual feature를 추가하면, 모델은 단순히 거리 크기만 보는 것이 아니라 거리 방정식의 모순을 관찰할 수 있다. 이 때문에 제한된 데이터에서도 baseline 대비 유의미한 오차 감소가 나타난 것으로 판단한다.

### 4.6 baseline 비교의 공정성

딥러닝 모델을 매우 단순한 삼각측량 또는 임의의 약한 알고리즘과 비교하면 공정하지 않을 수 있다. 본 보고서의 baseline은 제안 알고리즘의 실제 첫 단계인 inverse-distance weighted centroid이다. 즉, baseline과 GERT는 같은 입력 거리, 같은 MAD clipping, 같은 초기 위치 계산을 공유한다. 차이는 Transformer residual 보정기의 유무이다.

따라서 성능 개선은 단순히 전처리 효과 때문이 아니라, residual feature와 학습 기반 보정기가 추가되었을 때의 효과로 해석할 수 있다. 또한 validation set은 학습에 사용하지 않았으므로, 자체 평가는 hidden test set을 완전히 대체하지는 못하지만 과적합 여부를 확인하기 위한 최소한의 공정성을 갖는다.

### 4.7 장점, 한계, 향후 개선

본 알고리즘의 장점은 물리 기반 구조와 머신러닝을 결합했다는 점이다. 초기 위치를 통해 해석 가능한 출발점을 만들고, residual regression으로 보정량만 학습하기 때문에 좌표 직접 예측보다 과적합 위험을 낮출 수 있다. 또한 모델 크기를 9,778개 파라미터로 제한하여 10분 실행 제한에 대해 안정적인 추론이 가능하다.

한계도 존재한다. 첫째, 초기 weighted centroid가 크게 벗어나면 residual 자체가 왜곡되어 보정 성능이 제한될 수 있다. 둘째, validation set이 105개뿐이므로 split에 따라 metric 변동이 발생할 수 있다. 셋째, 본 프로젝트에서는 attention weight를 별도로 시각화하지 않았으므로 “어떤 기지국을 실제로 신뢰했는가”를 정량적으로 증명하지는 못했다. 넷째, WLS trilateration, robust least squares, Kalman filter와의 직접 비교는 수행하지 않았다.

향후에는 weighted centroid 대신 WLS 기반 초기 추정기를 사용하고, residual channel 제거 ablation, distance-only Transformer, MLP residual model을 추가 비교할 수 있다. 또한 여러 random seed에 대한 반복 검증을 수행하면 validation metric의 신뢰도를 더 높일 수 있다. 이동 궤적 데이터가 주어진다면 Kalman filter 또는 temporal Transformer와 결합하여 시간적 연속성을 반영하는 방향으로 확장할 수 있다.

## 5. Reference

[1] H. Liu, H. Darabi, P. Banerjee, and J. Liu, “Survey of Wireless Indoor Positioning Techniques and Systems,” IEEE Transactions on Systems, Man, and Cybernetics, Part C, vol. 37, no. 6, pp. 1067–1080, 2007.

[2] S. Gezici, Z. Tian, G. B. Giannakis, H. Kobayashi, A. F. Molisch, H. V. Poor, and Z. Sahinoglu, “Localization via Ultra-Wideband Radios: A Look at Positioning Aspects for Future Sensor Networks,” IEEE Signal Processing Magazine, vol. 22, no. 4, pp. 70–84, 2005.

[3] P. J. Rousseeuw and C. Croux, “Alternatives to the Median Absolute Deviation,” Journal of the American Statistical Association, vol. 88, no. 424, pp. 1273–1283, 1993.

[4] A. Vaswani et al., “Attention Is All You Need,” Advances in Neural Information Processing Systems, 2017.

[5] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” Journal of Computational Physics, vol. 378, pp. 686–707, 2019.
