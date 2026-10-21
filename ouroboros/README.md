# PROJECT OUROBOROS v2

단세포 아메바(파라미터 2,496개)에서 출발해 세포 분열로 팔을 늘리고, 다림질한 가시 산을 천장 유압 프레스로 굳히고,
파이썬 AST로 자기 유전자 코드를 고쳐 쓰는 무한 진화 루프입니다.

| 모듈 | 역할 |
|---|---|
| `OUROBOROS_01_energy_bedroom_and_spikes.py` | 안방 바닥 물리 엔진 + 가시 산 생성기 |
| `OUROBOROS_02_amoeba_morphogenetic_brain.py` | 세포 분열/자연사하는 형태발생 뇌 |
| `OUROBOROS_03_steam_iron_and_hydraulic_press.py` | 분신술 스팀다리미 + 천장 유압 프레스 컴파일러 |
| `OUROBOROS_04_ast_dna_self_coder.py` | Huxley-Gödel AST 자가 코딩 엔진 |
| `OUROBOROS_05_paranoid_nightmare_conjecturer.py` | 사각지대 악몽 자가 대국 |
| `OUROBOROS_06_butler_transducer_and_decoders.py` | 집사 통역기 (신약/수학/코드/대화) |
| `OUROBOROS_07_kaggle_infinite_daemon.py` | 체크포인트 이어받기 무한 진화 데몬 |
| `build_kaggle_kernel.py` | 7개 모듈을 Kaggle 스크립트 커널 하나로 묶는 발사대 |

## v2 업그레이드

1. **유전자 뷔페 무한 리필기**: `GeneMutator(ast.NodeTransformer)`가 상수 흔들기, `tanh/sin/arctan` 교체, `+/-` 뒤집기로
   새 유전자를 발명합니다. 그 유전자를 쓴 후보가 챔피언이 되면 유전자 은행에 영구 등록됩니다.
   현재 챔피언도 같은 시험으로 재채점받는 엘리트 보존이 들어 있고, 챔피언 계보는 `dna_lineage.jsonl`에 남습니다.
2. **족보 스위치**: `allow_oracle_peek`. 계기판은 족보 ON(원조 지표), 족보 OFF, 처음 보는 손님(unseen walk-in)
   세 가지로 동시에 채점합니다. 측정용 다림질에서는 세포 분열이 일어나지 않습니다.
3. **체크포인트 이원화**: `.pkl`과 `.pkl.bak`에 원자적으로 저장하고 rng 상태까지 보존합니다.
   `--resume-from` 폴더(Kaggle에서는 `/kaggle/input`)에서 가장 최신 체크포인트를 찾아 이어받습니다.
4. **악몽 × 셀프코더**: 최근 사냥한 사각지대 악몽들이 AST 유전자 적합도 평가셋으로 들어갑니다.
5. **통역사 상시 근무**: 매 에포크마다 처음 보는 손님 요청을 받아 족보 없이 다림질하고 브리핑을 `butler_guest_log.jsonl`에 남깁니다.
6. **장기 주행 대비**: 프레스 검색을 행렬곱으로 벡터화했고, 프레스 수 상한과 은퇴 정책을 두었습니다.
   계층 보너스는 L12에서 상한을 둡니다. `--hours` 시간 예산이 다 되면 스스로 정리하고 저장합니다.

## 실행

```bash
# 로컬
python OUROBOROS_07_kaggle_infinite_daemon.py --out runs/local --epochs 50

# Kaggle (토큰은 환경변수로만)
python build_kaggle_kernel.py --user <kaggle-user> --hours 11.0
KAGGLE_API_TOKEN=... kaggle kernels push -p kaggle_kernel

# 다음 세션에서 이어 진화: 이전 커널 출력을 입력으로 붙여서 다시 push
python build_kaggle_kernel.py --user <kaggle-user> --hours 11.0 --resume-kernel <kaggle-user>/ouroboros-infinite-evolution
```
