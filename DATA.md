# 아주소중한딥러닝챌린지_2026

## Dataset Description

### 데이터셋 구성
총 3개의 데이터셋이 제공됩니다.

- deep_chal_math_dataset_train.csv: 챌린지의 학습에 직접적으로 사용하셔야 할 데이터셋으로 train set으로 구성됩니다.
- deep_chal_math_dataset_leaderboard.csv: 실시간 리더보드 순위를 결정하는 평가 데이터셋입니다. answer 필드는 비워져 있습니다. (2026년 8월 3일 14:00부로 아래 정제 버전(deep_chal_math_dataset_leaderboard_filtered.csv)으로 대체되었습니다.)
- deep_chal_math_dataset_test.csv: 최종 순위를 평가하는 데이터셋으로, 8월 31일 00:00에 데이터셋이 공개됩니다. 2026.08.31 00:00 ~ 2026.08.31 23:59 내에 답안을 제출해주세요.

[26년 8월 3일 14:00 수정]
- deep_chal_math_dataset_leaderboard_filtered.csv: 기존 리더보드 평가 데이터셋에서 오류가 확인된 항목을 제거한 정제 버전입니다. 2026년 8월 3일 14:00부터 본 파일을 사용해 주시기 바랍니다. 항목 수가 변경되었으므로, 제출 파일은 반드시 정제 버전의 id를 기준으로 작성해 주시기 바랍니다.
- train_filtered_ids.csv: train 데이터셋에서 오류가 확인된 항목의 id 목록입니다. 학습 시 이를 참고하여 자체적으로 필터링을 적용해 주시기 바랍니다.

### 데이터 포맷
모든 데이터는 csv 형태로 통합되어 구성되어 있으며 다음 필드들을 포함합니다.

필드 이름 | 설명 |
id | 각 문항을 구분하기 위한 고유 식별자입니다. 제출 파일과 정답을 매칭하는 기준이 되므로 임의로 변경하시면 안 됩니다. |
question | 모델에 입력되는 수학 문제 텍스트입니다. 자연어 서술과 LaTeX 형태의 수식이 함께 포함될 수 있습니다. |
answer | 문제의 최종 정답입니다. 모든 정답은 정수 형태로 통일되어 있으며, Test 데이터셋에서는 비어 있습니다. |

### 예시

```
{
"id": "train-000000",
"question": "What is the molecular weight of some moles of Aluminum chloride if the molecular weight of 3 moles is 396?", 
"answer": "132"
}
```

---

### License
Apache 2.0
