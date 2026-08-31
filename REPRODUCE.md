# 최종 제출물 재현 가이드 (아주소중한딥러닝챌린지 2026)

제출한 `test_submission.csv`(answer 컬럼 채움본)는 아래 절차로 동일하게 재현됩니다.
모든 샘플링은 `seed=42`로 고정되어 있습니다.

## 1. 환경

- GPU 2장 (실험 환경: NVIDIA A6000 ×2, CUDA driver 11.8)
- 추론 환경: Python 3.11, `vllm==0.6.1.post2(+cu118)`, `transformers==4.45.2`, `torch==2.4.1+cu118`, `pandas`
- 학습(가중치 재생성 시에만): `transformers==4.55.4`, `trl==0.21.0`, `peft==0.17.0`, `accelerate==1.10.0`
- 자동 셋업: `bash baseline/scripts/00_setup_server_envs.sh` (경로는 `baseline/scripts/env_server.sh`에서 수정)

## 2. 모델 가중치

베이스 모델은 대회 지정 `Qwen/Qwen2.5-3B-Instruct`(HuggingFace 자동 다운로드)이며,
fine-tuned LoRA 어댑터는 본 저장소 **Releases**에 첨부되어 있습니다.

- `lora-m3.zip` — 최종 앙상블 멤버 M3 (필수)
- `lora-m1.zip` — 백업 단일모델 M1 (스크립트가 함께 실행)
- (`lora-m2.zip`, `lora-m4.zip` — 실험 과정 산출물, 최종 제출에는 미사용)

압축을 풀어 `$DLC_OUTPUT_DIR/lora-m3` 등으로 두고, 베이스와 병합:

```bash
python src/merge_lora.py --adapter "$DLC_OUTPUT_DIR/lora-m3" --out "$DLC_OUTPUT_DIR/merged-m3"
python src/merge_lora.py --adapter "$DLC_OUTPUT_DIR/lora-m1" --out "$DLC_OUTPUT_DIR/merged-m1"
```

## 3. 최종 추론 (제출 CSV 재현)

```bash
cd baseline
conda activate dlc
source scripts/env_server.sh
bash scripts/20_final_test.sh "$DLC_DATA_DIR/test_submission.csv"
# 산출: final_test/submission_ensemble.csv (id,answer)

python src/fill_answers.py \
    --template "$DLC_DATA_DIR/test_submission.csv" \
    --answers  "$DLC_OUTPUT_DIR/final_test/submission_ensemble.csv" \
    --out      "$DLC_OUTPUT_DIR/final_test/test_submission_FILLED.csv"
```

최종 파이프라인 구성: base 모델과 M3를 각각 (k=16, T=0.8)과 (k=32, T=0.6)로 self-consistency
샘플링(총 96표/문항) → 동률 가중 다수결 → 합의율 45% 이하의 불확실 문항만 두 모델 k=64로
집중 재샘플링(적응형 예산 배분) → 최종 다수결로 정수 답 확정.

## 4. 학습 재현 (어댑터를 처음부터 다시 만들 경우)

- 데이터 정제: `src/prepare_data.py` — 주최측 `train_filtered_ids.csv` + 커뮤니티 검증 오류
  리포트(ill-posed 602건 제거, 라벨 440건 교정) 자동 적용, 이후 모델 합의 기반 자체 라벨
  감사(`src/audit_labels.py`)로 잔여 오류 제거
- M1: `scripts/12_round2_m2.sh`의 M1 단계 — 베이스 모델 rejection sampling(STaR, k=8) 후
  정답 일치 풀이만으로 LoRA SFT (r=32, lr 1e-4, 2 epochs)
- M3: `scripts/13_sweep_and_m3.sh` — M1 데이터 + OpenMathInstruct-2의 GSM 계열 서브셋
  2만 건 혼합, LoRA SFT (r=32, lr 5e-5, 1 epoch)

## 5. 사용 데이터 출처 (전체 목록)

1. 주최측 제공: `deep_chal_math_train.csv`, `train_filtered_ids.csv`
2. 커뮤니티 공개 검증 리포트(운영진 인정): `organizer_report_mislabel_442.csv`,
   `organizer_report_illposed_623.csv`
3. 외부 공개 데이터(M3 학습에만): `nvidia/OpenMathInstruct-2` (HuggingFace, Apache-2.0)
   중 `problem_source`가 GSM 계열인 행만 필터링(`src/prepare_external_data.py`), 2만 건
4. 자체 생성 데이터: 위 모델들의 rejection sampling 풀이 (베이스 모델 산출물)

상용 API 미사용, 추론 시 외부 호출 없음(오프라인 vLLM).
