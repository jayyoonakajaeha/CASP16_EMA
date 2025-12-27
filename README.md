# CASP16 EMA 평가 패키지 (Evaluation Package)

이 패키지는 CASP16 공식 문서 및 방법론에 기반하여 단백질 구조 예측 모델의 품질(QA)을 평가하고 랭킹을 산출하는 도구 모음입니다.

## 📌 주요 기능

1.  **2-Pass Outlier Removal Z-Score**: 이상치에 강건한 Z-score 산출 알고리즘 적용 (Outlier 제거 후 평균/표준편차 재계산).
2.  **CASP16 Ranking Score (RS)**: `0.5 * Z(Pearson) + 0.5 * Z(Spearman) + Z(AUROC) + Z(Loss)` 공식을 통한 종합 점수 계산.
3.  **Strict Filtering**: 
    *   **Low Quality Target 제외**: 최고 성능 모델의 TM-score가 0.6 미만인 타겟은 변별력 부족으로 평가에서 제외.
    *   **Data Coverage Rule**: 타겟별 예측 제출율이 80% 미만인 모델은 해당 타겟 점수 0점 처리.
4.  **Robust Metrics**:
    *   **Loss**: Best Possible Score와의 차이 (Lower is Better).
    *   **AUROC**: 타겟 내 **상위 25% (75th percentile)** 모델을 'Good'으로 정의하여 변별력 측정 (상대평가).

## 📂 패키지 구조

```
CASP16_Evaluation_Pkg/
├── grade_casp16_predictions.py  # [Step 1] 원본 예측 데이터를 정답과 비교하여 4가지 지표(P, S, L, A) 산출
├── calculate_casp16_zscores.py  # [Step 2] 산출된 지표를 바탕으로 Z-Score 및 최종 RS 랭킹 계산
├── merge_predictions.py         # (선택) 여러 예측 파일 합치기 도구
├── targetlist.csv               # CASP16 타겟 리스트 (참고용)
└── requirements.txt             # 필요 라이브러리 목록
```

## 🚀 사용 방법 (Quick Start)

### 1. 환경 설정

Python 3.8 이상 환경에서 필요한 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

*   **Prediction Files**: 타겟별 예측 점수가 담긴 CSV 파일들 (예: `T1104.csv`, `H1202.csv`). 각 파일은 EMA 모델들을 컬럼으로 가져야 합니다.
*   **Quality Score Files**: 실제 정답 품질 점수(GDT-TS, TM-score 등)가 담긴 CSV 파일들 (예: `T1104_quality_scores.csv`).

### 3. [Step 1] 1차 채점 (Grading)

각 예측 모델의 성능 지표(Pearson, Spearman, Loss, AUROC)를 계산합니다. 이 단계에서 **타겟 필터링(TM-score < 0.6 제외)**과 **AUROC 기준 설정(Top 25%)**이 적용됩니다.

```bash
python grade_casp16_predictions.py \
  --pred_dir ./data/Predictions \
  --score_dir ./data/Quality_Scores \
  --output graded_metrics.csv \
  --truth_metric tmscore_mmalign
```

*   `--pred_dir`: 예측 파일들이 있는 디렉토리 (패키지 내 `data/Predictions`에 예제 데이터 포함됨)
*   `--score_dir`: 정답 파일들이 있는 디렉토리 (패키지 내 `data/Quality_Scores`에 예제 데이터 포함됨)
*   `--output`: 결과가 저장될 CSV 경로
*   `--truth_metric`: (선택) 정답으로 사용할 CSV 컬럼명. 기본값은 `tmscore_mmalign`입니다. `tmscore_usalign`, `lddt` 등으로 변경하여 다른 지표 기준으로 평가할 수 있습니다.

### (옵션) 나만의 모델 추가하여 랭킹 비교하기

자신이 개발한 모델의 성능을 CASP16 리더보드에 포함시켜 비교할 수 있습니다.

1.  **예측 파일 준비**: 다음과 같은 형식의 CSV 파일을 준비합니다 (예: `my_predictions.csv`).
    *   필수 컬럼: `Target`, `Model` (Decoy ID), `Score`
    ```csv
    Target,Model,Score
    H1202,H1202TS014_1,0.85
    H1202,H1202TS014_2,0.82
    T1206,T1206TS475_1,0.77
    ...
    ```
    (주의: `Model` ID는 `Predictions` 디렉토리 내의 파일들에 있는 ID 형식과 일치해야 매칭됩니다.)

2.  **실행**: `--user_csv`와 `--user_name` 옵션을 추가하여 실행합니다.

```bash
python grade_casp16_predictions.py \
  ... (기존 옵션) \
  --user_csv my_predictions.csv \
  --user_name "My_Awesome_Model"
```

이렇게 하면 `graded_metrics.csv`에 "My_Awesome_Model"의 성적표가 포함되며, 이후 `calculate_casp16_zscores.py`를 실행할 때 자동으로 리더보드에 랭킹이 산출됩니다.

### 4. [Step 2] 2차 랭킹 산출 (Ranking)

1차 채점 결과를 바탕으로 2-Pass Z-score를 계산하고 최종 순위를 매깁니다.

```bash
python calculate_casp16_zscores.py \
  --input graded_metrics.csv \
  --output_dir ./results \
  --tm_metric tmscore_mmalign
```

*   `--input`: 1차 채점 단계에서 생성된 CSV 파일.
*   `--tm_metric`: 평가에 사용할 메트릭 그룹. `grade_casp16_predictions.py`에서 `--truth_metric`으로 지정한 값(기본 `tmscore_mmalign`)과 일치해야 합니다.
*   이 옵션을 사용하면 `SCORE` (Global Quality) 리더보드가 생성됩니다.

## 📊 결과 해석

생성된 `leaderboard_SCORE.csv` 파일 예시:

| Model | RS_SCORE | Rank |
| :--- | :--- | :--- |
| Model_A | 55.2 | 1 |
| Model_B | 48.1 | 2 |
| ... | ... | ... |

*   **RS_SCORE**: 각 타겟별 Z-score의 총합. 높을수록 좋습니다.
*   점수가 0.0인 경우: 예측을 제출하지 않았거나, 모든 타겟에서 이상치(Outlier)로 분류되어 제거되었거나, 타겟 필터링에 의해 유효한 평가 타겟이 없는 경우일 수 있습니다.

---
**Last Updated**: 2025-12-27
