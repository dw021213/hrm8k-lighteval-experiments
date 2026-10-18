# HRM8K 채점 실험 (B팀 2주차 분석용)

분석 보고서(`teams/team-b-hrm8k/analysis.md`)에 인용한 숫자를 다시 만들 수 있게 모아 둔 실험 코드다.
팀 구현 코드가 아니며, 운영 저장소나 Lighteval 저장소에 올리지 않는다.

## 실행 환경

- `../lighteval/.venv` (Python 3.12, lighteval 0.13.1.dev0 @ `c2af9e5`)
- 데이터: `HAERAE-HUB/HRM8K` @ `c360cabf8d733a82455565358b3dc965aab9ba8d`

```bash
cd ~/Desktop/오픈소스/hrm8k-experiments
PY=python   # lighteval @ c2af9e5 가 설치된 환경
$PY e1_load.py      # 데이터 스키마: config, split, 컬럼, answer 자료형
$PY e2_perfect.py   # 정답을 그대로 \boxed{}에 쓴 '완벽한 답'을 채점기가 몇 % 인정하는지
$PY e3_variants.py  # 같은 정답을 다르게 쓴 답(-8 \pi, 0.5, 2,880,000, 4번 ...)을 채점기별로 비교 + Language.KOREAN 오류 재현
$PY e5_known_issues.py  # 제보할 KSM 오류 문항(3, 109, 201, 323번 행)과 82번 행을 다시 계산
```

연습용 태스크를 실제 명령으로 돌리려면:

```bash
# lighteval @ c2af9e5 가 설치된 환경에서
lighteval accelerate "model_name=HuggingFaceTB/SmolLM2-135M-Instruct,batch_size=1" "hrm8k_proto|0" \
  --custom-tasks hrm8k_proto.py --max-samples 2 --output-dir ./results/proto --save-details
```

## 파일

| 파일 | 내용 |
|---|---|
| `scorers.py` | 비교한 채점기 6개. lm-eval `hrm8k`, Lighteval `expr_gold_metric`, 정답을 LaTeX로 읽는 변형 3개, 분석 보고서 5-4의 제안 규칙(`proposed_rule`) |
| `third_party/lmeval_hrm8k_utils.py` | lm-evaluation-harness `hrm8k` 채점 코드 사본 (MIT, 커밋 `314f717`) |
| `hrm8k_proto.py` | 실행 가능성 확인용 연습 태스크. 프롬프트는 lm-eval 한국어판, 정답은 `$…$`로 감싸 LaTeX·식으로 읽음 |
| `hrm8k_mc_proto.py` | MMMLU를 선택지 확률(log-likelihood)로 채점하는 연습 태스크. SmolLM2-135M 20문항 acc 0.45 (`results/mc_proto.log`) |
| `e5_known_issues.py` | 데이터 오류 제보 근거 재계산(수치 적분, 순열 세기, 나선 예시) |
| `results/` | 2026-10-16 실행 결과 |

## 결과 요약 (2026-10-16)

완벽한 답 인정률 (`results/e2_perfect.txt`):

| 묶음 | lm-eval hrm8k | Lighteval expr_gold | 제안 규칙 (`proposed_rule`) |
|---|---|---|---|
| GSM8K | 99.8% | 100% | 100% |
| MATH | 100% | 100% | 100% |
| OMNI_MATH | 100% | 100% | 100% |
| MMMLU | 100% | 100% | 100% |
| KSM | 83.6% | 97.1% | 99.9% |

- lm-eval은 정답 쪽에도 `parse_math_answer`를 적용해서 `-10`→`10`, `\frac{1}{2}`→`2`처럼 바뀐다. 이 때문에 236문항(GSM8K 2, KSM 234)은 완벽한 답도 정답 처리되지 않는다.
- 제안 규칙이 놓친 1문항은 KSM[201]이고, 데이터에 `\frac[13}{20}`(괄호 오타)로 저장돼 있다.
- 이 실험은 '정답 자체'를 답으로 넣은 것이라, 정답이 틀린 KSM[3], KSM[109]는 여기서 드러나지 않는다. 두 문항은 `e5_known_issues.py`로 확인한다.
