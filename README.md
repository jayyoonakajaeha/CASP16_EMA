# CASP16 EMA 모델 평가 패키지 (Z-Score 2단계 알고리즘)

이 패키지는 **CASP16의 Z-Score 평가 방법론**을 사용하여 단백질 구조 모델(EMA)을 평가하기 위한 독립적인 도구 모음입니다.

## 주요 기능

1.  **CASP16 방법론 정밀 구현**:
    *   **2단계 아웃라이어 제거 (2-Pass Outlier Removal)**: 평균에서 2 표준편차 이상 벗어난 '나쁜' 모델을 제거한 후 통계를 다시 계산하여 Z-Score를 산출합니다.
    *   **타겟별 합산 (Target-wise Summation)**: 각 타겟별로 Z-Score를 계산한 뒤, 이를 모두 합산하여 최종 순위를 결정합니다.
    *   **음수 클램핑 (Negative Clamping)**: 평균보다 낮은 성능(음수 Z-Score)은 0.0점으로 처리하여 과도한 페널티를 방지합니다.
    *   **방향성 처리 (Directionality Handling)**: '낮을수록 좋은' 지표(예: Loss, RMSD)에 대해서는 Z-Score 방향을 자동으로 보정합니다.

## 디렉토리 구조

```
CASP16_Evaluation_Pkg/
├── calculate_casp16_zscores.py  # 메인 Z-Score 계산 스크립트
├── merge_predictions.py         # 개별 CSV 파일 병합 도구
├── requirements.txt             # Python 의존성 목록
└── README.md                    # 설명서 (본 파일)
```

## 설치 방법

1.  Python 3를 설치합니다.
2.  의존성 라이브러리를 설치합니다:
    ```bash
    pip install -r requirements.txt
    ```

## 사용 방법

### 단계 0: 원본 데이터 채점 (옵션 - 80% 규칙 적용)

만약 예측한 Raw Score (모델별 점수) 파일들을 가지고 있다면, 이 단계부터 시작하세요.
이 스크립트는 CASP16의 엄격한 **'80% 데이터 충족 요건'**을 검사하고, 통과한 타겟에 대해서만 지표(Pearson, Loss 등)를 계산합니다.

```bash
python grade_predictions.py \
  --pred_dir /path/to/raw_predictions \
  --score_dir /path/to/ground_truth \
  --output graded_metrics.csv \
  --metric_pairs RF_tmscore:tm_score Stacking_lddt:lddt \
  --directions higher_is_better higher_is_better
```
*   `--metric_pairs`: 예측 컬럼과 정답 컬럼을 `PredictorCol:TruthCol` 형태로 매핑합니다.
*   **80% 규칙**: 타겟 내 모델의 80% 이상을 예측하지 못한 경우, 해당 타겟의 모든 지표는 `NaN` 처리되어 이후 0점(Z-Score)이 됩니다.

### 단계 1: 예측 데이터 준비 (이미 채점된 경우)

만약 이미 채점된 메트릭(Pearson 등) 파일이 있다면 이 단계로 건너뛰세요.
모든 타겟과 모델의 예측 결과가 포함된 하나의 통합 CSV 파일이 필요합니다.

CSV 파일은 반드시 다음 컬럼들을 포함해야 합니다:
*   `Target`: 타겟 ID (예: T1104, T1106s1)
*   `Model`: 모델명 (예: MULTICOM, AlphaFold2)
*   `Metric1`, `Metric2`, ...: 수치형 점수 (예: tm_score, lddt, dockq, rmsd)

만약 예측 결과가 타겟별로 개별 CSV 파일(예: 폴더 내 `T1104.csv`)에 흩어져 있다면, 아래의 병합 스크립트를 사용하세요:

```bash
python merge_predictions.py \
  --pred_dir /path/to/predictions_directory \
  --output merged_predictions.csv
```

### 단계 2: Z-Score 평가 실행

평가 스크립트를 실행할 때 입력 파일, 평가할 지표 목록, 그리고 각 지표의 방향성을 지정해야 합니다.

**사용 예시:**
```bash
python calculate_casp16_zscores.py \
  --input merged_predictions.csv \
  --output leaderboard \
  --metrics tm_score lddt dockq rmsd \
  --directions higher_is_better higher_is_better higher_is_better lower_is_better \
  --threshold 2.0
```

*   `--input`: 1단계에서 준비한 병합된 CSV 파일 경로.
*   `--metrics`: 평가할 지표(컬럼명)들을 공백으로 구분하여 입력.
*   `--directions`: 각 지표에 해당하는 방향성 (`higher_is_better` 또는 `lower_is_better`)을 순서대로 입력.
*   `--threshold`: 아웃라이어 제거를 위한 Z-Score 임계값 (기본값: 2.0).

### 출력 결과

스크립트 실행 후 다음과 같은 파일들이 생성됩니다:
1.  **지표별 리더보드 CSV** (예: `leaderboard_tm_score.csv`, `leaderboard_rmsd.csv`).
2.  각 리더보드 파일의 구성:
    *   `Model`: 모델명
    *   `Summed_Z_Score`: 최종 합산 점수 (높을수록 좋음)
    *   `Rank`: 순위 (1위부터 시작)

## 라이선스

MIT License.
