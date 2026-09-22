# 6뷰 → 단일 메쉬: 통합 병목의 구조적 해결

> 상태: 기획. 코드 없음. 이 문서가 합의되면 구현 시작.

## 0. 진단 — 왜 터졌는가

관측된 증상: 뷰별 2.5D 퀄리티는 훌륭한데 통합하면 메쉬가 튀고, 뭉개지고,
날아가고, 찌그러짐. **OBJ를 직접 6뷰 렌더한 GT 이미지로도 실패.**

마지막 문장이 이 프로젝트에서 가장 중요한 데이터입니다. GT 입력으로도 실패했다면
원인은 "생성 이미지가 흔들려서"가 **아닙니다.** 파이프라인의 정식화 자체가 틀렸다는
뜻이에요. 이미지 품질을 아무리 올려도 안 고쳐집니다.

### 근본 원인 4개

**(1) 깊이 융합은 "다수결"인데 투표자가 없다.**
TSDF 융합(Curless & Levoy 1996)은 KinectFusion류에서 잘 돕니다. 표면의 한 점을
수백 프레임이 관측하기 때문이에요. 이상치는 평균에 묻힙니다.
직교 6뷰에서는 표면의 한 점을 관측하는 뷰가 **1개, 많아야 2개**입니다.
중복이 0이에요. **나쁜 깊이 픽셀 하나가 곧 그 지점의 표면이 됩니다.**
이상치 모델이 없는 상태에서 통계적 중복에 의존하는 방법을 쓴 것 — 이게 1번.

**(2) 단안 깊이는 뷰마다 scale·shift가 미지수다.**
Marigold, MiDaS, Depth Anything 계열은 전부 *affine-invariant* 깊이를 냅니다
(Ranftl et al., MiDaS). 출력이 `a·d + b` 만큼 자유롭고 a, b는 뷰마다 다릅니다.
보정 없이 6장을 back-project 하면 6개의 점구름이 **서로 다른 깊이 구간에 따로**
떠 있습니다. 교집합이 없으니 융합할 게 없고, 강제로 융합하면 모순된 부호가
생겨 zero-crossing이 떠돌아다닙니다. → **"메쉬가 날아감"의 정체.**

**(3) 실루엣 경계의 flying pixel.**
깊이 불연속 지점에서 back-projection은 전경과 배경을 잇는 스커트를 만듭니다.
이게 그대로 삼각형이 되면 카메라 방향으로 길게 뻗은 스파이크가 됩니다.
→ **"메쉬가 튐"의 정체.** 융합 알고리즘 문제가 아니라 전처리 누락입니다.

**(4) 한 추정기에 저주파와 고주파를 동시에 요구했다.**
깊이망은 전역 형상(저주파)엔 강하고 표면 문양(고주파)엔 약합니다. 이미지 미분
(Scharr)은 정반대로 고주파만 있고 저주파는 전혀 없습니다. 둘을 하나의 깊이맵으로
합치려 하면 문양은 뭉개지고 형상은 울퉁불퉁해집니다.
→ **"뭉개짐"의 정체.** 주파수 대역을 분리하지 않은 것.

추가로, 생성 이미지의 드리프트(원인 #1로 지목하신 것)는 실재하지만 **부차적**입니다.
위 4개를 고치면 드리프트는 잔차로 흡수됩니다. 안 고치면 GT로도 실패합니다 —
이미 실험으로 증명하셨고요.

---

## 1. 설계 원칙 5개

### 원칙 1 — 평균내지 말고 가두어라 (Constrain, don't average)

관측 중복이 없으면 **제약과 사전지식**으로 대체해야 합니다. 통계 대신 기하학.

6장의 실루엣에서 **visual hull**(Laurentini 1994; Kutulakos & Seitz, Space Carving)을
공간 조각으로 만듭니다. 6뷰 hull은 뚱뚱하고 오목한 부분을 못 잡습니다. 그래도 됩니다.
최종 메쉬로 쓸 게 아니거든요. hull의 역할은 딱 하나:

> **실제 표면은 hull 안에 있음이 수학적으로 보장된다.**

이걸 최적화의 **하드 상한**으로 씁니다. Poisson 재구성은 envelope 제약을 지원합니다
(Kazhdan et al. 2020, *Poisson Surface Reconstruction with Envelope Constraints*).
SDF 최적화라면 hull 바깥에 페널티를 겁니다.

**메쉬는 우리에 갇혀 있으면 날아갈 수 없습니다.** 원인 (2)에 대한 구조적 봉쇄.

그리고 실루엣은 이 파이프라인 전체에서 **가장 신뢰도 높은 신호**입니다. 깊이망보다
정확하고, 생성 드리프트에 덜 민감하고, 공짜예요. 지금까지 안 쓰고 있었다면 그게
제일 큰 낭비입니다.

### 원칙 2 — 드리프트를 싸우지 말고 변수로 풀어라

이게 이 기획의 핵심입니다.

지금까지의 접근: "6장을 완벽히 일관되게 만든 뒤 융합한다" → 불가능. 생성모델은
절대 픽셀 일관성을 안 줍니다. GT 렌더로도 안 됐다는 건 이 전제 자체가 틀렸다는 뜻.

바꿀 정식화: **불일치를 자유 파라미터로 모델링하고 형상과 함께 동시에 최적화한다.**

뷰마다 다음을 미지수로 둡니다:
- 2D 유사변환 (scale, tx, ty, 소각도 회전) — 생성/렌더 정렬 오차 흡수
- 깊이 affine (a_i, b_i) — 원인 (2) 정면 해결
- (선택) 저차 warp field — 국소 형상 드리프트 흡수, 크게 정규화

그리고 형상 파라미터와 **같은 손실함수 안에서 같이** 내립니다. 고전 bundle
adjustment랑 같은 구조예요. 카메라 자세 대신 "생성 드리프트"를 nuisance
parameter로 둔 것.

> 불일치는 제거해야 할 오류가 아니라, 추정해야 할 변수입니다.

이렇게 하면 이미지가 흔들려도 시스템이 무너지지 않습니다. 잔차가 조금 커질 뿐이에요.

### 원칙 3 — 주파수 대역을 분리하라

| 대역 | 출처 | 산출물 |
|------|------|--------|
| 저주파 (전체 형상, 부피) | visual hull + 보정된 깊이 | 베이스 메쉬 |
| 중주파 (판 구조, 근육 덩어리) | normal map 적분 | 정제된 메쉬 |
| 고주파 (판 이음새, 육각 균열, 문양) | Scharr / 이미지 미분 | displacement + normal map |

**깊이가 아니라 normal을 1차 신호로 쓰는 것**이 중요합니다. 이유:
- normal은 scale/shift 모호성이 없습니다 (원인 2가 애초에 발생 안 함)
- 고주파 표면 디테일을 깊이보다 훨씬 잘 담습니다
- 직교/원근 카메라 차이에 덜 민감합니다

normal → 표면 복원은 **normal integration**으로 합니다.
Cao et al. 2022, *Bilateral Normal Integration* (BiNI) 그리고 그 깊이 정규화
버전인 **d-BiNI**.

### 원칙 4 — 대칭은 공짜 데이터다

Cinder Basilisk는 완전 좌우대칭으로 설계했습니다. 그러면 형상에 하드 제약을 겁니다:

```
f(x, y, z) = f(-x, y, z)
```

효과:
- 좌·우 측면 뷰가 **서로를 구속**합니다. 관측 중복이 실질 2배.
- 비대칭 아티팩트가 표현 자체에서 불가능해집니다.
- 자유도가 절반으로 줄어 최적화가 안정됩니다.

원인 (1)의 "투표자가 없다"에 대한 가장 값싼 반격입니다. 관련: Wu, Rupprecht,
Vedaldi 2020, *Unsupervised Learning of Probably Symmetric Deformable 3D Objects*.

### 원칙 5 — GT 렌더를 단위 테스트로 삼아라

이미 GT 렌더로 실패하는 걸 발견하셨습니다. 그럼 그걸 **버그가 아니라 테스트 하네스로**
쓰면 됩니다.

알려진 메쉬(Stanford dragon, Blender 크리처 등)를 6뷰 직교 렌더 → 파이프라인 →
원본과 Chamfer distance / normal consistency / volumetric IoU 비교.

**이 테스트를 통과하기 전에는 생성 이미지를 넣지 않습니다.** 지금까지 디버깅이 힘들었던
이유는 "이미지가 나쁜 건지 파이프라인이 나쁜 건지" 구분이 안 됐기 때문이에요.
GT 테스트가 그 변수를 제거합니다. 숫자가 나오면 개선인지 아닌지 알 수 있고요.

---

## 2. 파이프라인

```
[6 PNG]
   │
   ├─ S1  분할 → 알파 마스크 (BiRefNet / SAM 2)
   │
   ├─ S2  정준화: 뷰 간 bbox 일관성 최소제곱
   │        front/back/top/bottom 폭 일치
   │        left/right/top/bottom 길이 일치
   │        front/back/left/right 높이 일치
   │        → 6개 유사변환 초기값 (S6에서 계속 최적화됨)
   │
   ├─ S3  Visual hull: 6 실루엣 공간 조각 → 복셀 점유격자
   │        + 대칭 강제
   │        → 하드 외피 (이후 전 단계의 상한)
   │
   ├─ S4  뷰별 추정
   │        normal  ← Marigold-Normals / StableNormal / DSINE   (주력)
   │        depth   ← Marigold / Depth Anything V2 / DA3        (보조)
   │        edge    ← Scharr / Sobel 고주파 밴드
   │
   ├─ S5  깊이 affine 보정 + flying pixel 제거
   │        hull을 직교 렌더 → 뷰별 hull 깊이 h_i
   │        robust fit: a_i·d_i + b_i ≈ h_i   (Huber, 내부 영역만)
   │          ※ hull 깊이는 편향된 상한이므로 스케일 앵커로만 쓰고
   │            타겟 기하로는 쓰지 않음
   │        |∇d| 임계 초과 픽셀 폐기 + 마스크 침식 → 스커트 제거
   │
   ├─ S6  ★ 통합 = 공동 역최적화  (여기가 전부)
   │        표현: FlexiCubes 또는 hash-grid neural SDF
   │        미지수: 형상 θ + 뷰별 {유사변환, depth affine, warp}
   │        손실:
   │          L_sil     실루엣 IoU        (미분가능 렌더, 하드에 가깝게 가중)
   │          L_normal  normal 일치       (주력 기하 신호)
   │          L_depth   보정 깊이, Huber   (보조, 낮은 가중)
   │          L_hull    hull 외부 페널티   (하드)
   │          L_eik     |∇f| = 1          (SDF 폭주 방지)
   │          L_lap     Laplacian 평활     (Nicolet 전처리)
   │          L_drift   뷰 파라미터 정규화 (드리프트가 커지지 않게)
   │        + 대칭 하드 제약
   │
   ├─ S7  메쉬 추출: marching cubes / FlexiCubes → 다양체 정리 → 리메쉬
   │
   └─ S8  디테일 베이킹: Scharr 고주파 → 뷰 투영 →
            normal map + displacement (세분화 메쉬에 적용)
          → 하이폴리 최종
```

### 왜 이게 안 터지는가 — 증상별 대응표

| 기존 증상 | 대응 장치 |
|-----------|-----------|
| 메쉬가 날아감 | hull 외피 하드 제약 (S3/S6) + depth affine 보정 (S5) |
| 스파이크가 튐 | flying pixel 제거 (S5) + 암시적 표현이라 스커트 삼각형 자체가 없음 |
| 뭉개짐 | 주파수 분리 (S4/S8) — 디테일을 깊이에서 뽑지 않음 |
| 찌그러짐 | 대칭 하드 제약 + 실루엣 손실이 전역 비율을 잡음 |
| 뷰마다 다른 생물 | 드리프트를 변수로 흡수 (S6) — 일관성을 요구하지 않음 |
| 구멍 | 암시적 표현은 정의상 watertight |
| 최적화 발산 | eikonal + Laplacian 전처리 (Nicolet et al. 2021) |

---

## 3. 정직한 한계

**6뷰가 못 보는 영역이 반드시 존재합니다.** 다리 6개짜리 생물은 실루엣 기법
최악의 케이스예요:
- 측면에서 중간 다리는 앞다리 뒤에 완전히 가려짐
- 다리 사이 배 아래쪽은 **어느 뷰에서도 안 보임** (관측 0회)
- 왕관 뿔 7개는 정면에서 서로 겹침

관측이 0인 영역은 어떤 알고리즘도 복원 못 합니다. 사전지식이 채웁니다 (평활 + 대칭).
결과적으로 **몸통·머리·꼬리는 좋게, 다리는 서로 붙거나 물갈퀴처럼 나올 가능성이
높습니다.**

해결책 두 가지:
- **(a) 그대로 가고 S7에서 다리를 분리** — 스펙 준수. hull 조각에서 다리 사이
  공간을 못 파낸 거라 후처리로 어느 정도 가능.
- **(b) 3/4 앵글 2~4장 추가** — 아키텍처는 이미 N뷰입니다. 6은 하드코딩 아님.
  전후좌우 45° 4장 추가하면 다리 모호성이 거의 사라집니다.
  "6뷰" 스펙은 유지하고 **보조 뷰**로 취급하면 됩니다.

제 추천은 (a)로 먼저 돌려서 숫자를 보고, 다리가 실제로 뭉치면 (b). 결정은 넘깁니다.

### 생성 중이신 이미지에 대한 즉시 반영 사항

- **직교 유지하세요.** 정렬 가치가 더 큽니다. 깊이망이 원근 사진으로 학습돼서
  직교 입력이 OOD인 건 맞지만, 주력 신호를 normal로 두고 깊이는 affine 보정 후
  보조로만 쓰기 때문에 영향이 제한됩니다.
- **마스크가 깨끗하게 뽑히는 게 제일 중요합니다.** 배경 #808080 균일, 바닥 그림자
  없음 — 프롬프트 팩에 이미 반영돼 있습니다. 실루엣이 이 파이프라인의 척추예요.
- 위/아래 뷰의 **방향 규약**(주둥이가 위쪽 가장자리)을 꼭 지키세요. S2 정준화가
  이걸 전제합니다.

---

## 4. 논문

### 왜 터졌는지 설명하는 것
- Curless & Levoy 1996 — *A Volumetric Method for Building Complex Models from
  Range Images*. TSDF 융합의 원전. 중복 관측을 전제한다는 점이 핵심.
- Ranftl et al. 2020 — *MiDaS*. affine-invariant 깊이와 scale-shift 손실.
  원인 (2)의 이론적 출처.

### 가두기
- Laurentini 1994 — *The Visual Hull Concept for Silhouette-Based Image
  Understanding*.
- Kutulakos & Seitz 2000 — *A Theory of Shape by Space Carving*.
- Kazhdan & Hoppe 2013 — *Screened Poisson Surface Reconstruction*.
- Kazhdan et al. 2020 — *Poisson Surface Reconstruction with Envelope
  Constraints*. ← hull을 하드 제약으로 넣는 구체적 방법.

### 통합 (핵심 참고)
- **Xiu et al. 2023 — *ECON: Explicit Clothed humans Optimized via Normal
  integration* (CVPR).** 이 프로젝트와 구조적으로 같은 문제입니다. 앞/뒤 normal map
  2장을 d-BiNI로 적분해 coherent한 앞뒤 표면을 만들고, 암시적 형상 완성 +
  Poisson으로 봉합. **6뷰로 일반화하는 게 이 기획의 뼈대.**
- Cao et al. 2022 — *Bilateral Normal Integration* (ECCV). BiNI 원전.
- Wang et al. 2021 — *NeuS*. Yariv et al. 2021 — *VolSDF*.
- Li et al. 2023 — *Neuralangelo*. 고주파 기하 복원.
- Gropp et al. 2020 — *Implicit Geometric Regularization* (eikonal 항).

### 최적화가 폭주하지 않게
- **Nicolet et al. 2021 — *Large Steps in Inverse Rendering of Geometry*
  (SIGGRAPH Asia).** 정점 위치에 그냥 경사하강 하면 왜 터지는지, Laplacian
  전처리로 어떻게 고치는지. "메쉬 폭발"에 정확히 대응하는 논문.
- Shen et al. 2021 — *DMTet*. Shen et al. 2023 — *FlexiCubes*.
  최적화 가능한 메쉬 표현. FlexiCubes가 현재 최선.
- Munkberg et al. 2022 — *nvdiffrec*. Laine et al. 2020 — *nvdiffrast*.

### 사전지식
- Wu, Rupprecht, Vedaldi 2020 — *Unsupervised Learning of Probably Symmetric
  Deformable 3D Objects from Images in the Wild* (CVPR best paper). 대칭 사전지식.
- Lorensen & Cline 1987 — *Marching Cubes*.

### 뷰별 추정
- Ke et al. 2024 — *Marigold: Repurposing Diffusion-Based Image Generators for
  Monocular Depth Estimation* (CVPR oral). + Marigold-Normals.
- Yang et al. 2024 — *Depth Anything V2*.
- Depth Anything 3 (2025) — 다중뷰 입력.
- Bae & Davison 2024 — *DSINE* (surface normal 추정).
- Ravi et al. 2024 — *SAM 2*. Zheng et al. 2024 — *BiRefNet* (고해상 분할).

---

## 5. 모델 / 라이브러리 목록

3D 생성 메쉬 모델(Tripo, Hunyuan3D, InstantMesh, TRELLIS 등) 미사용. 전부
"추정 + 최적화"이고 형상은 우리 손실함수가 만듭니다.

### 다운로드할 모델

| 역할 | 모델 | 비고 |
|------|------|------|
| 분할 | **BiRefNet** (`ZhengPeng7/BiRefNet`) | 고해상 매팅, 실루엣 주력 |
| 분할 백업 | SAM 2.1 Hiera-L | 프롬프트 기반 보정용 |
| **Normal** | **Marigold-Normals** (`prs-eth/marigold-normals-v1-1`) | **1차 기하 신호** |
| Normal 대안 | StableNormal / DSINE | 앙상블 비교용 |
| Depth | Marigold v1-1 (`prs-eth/marigold-depth-v1-1`) | 보조, affine 보정 후 |
| Depth | Depth Anything V2 Large | 속도/앙상블 |
| Depth | Depth Anything 3 | 다중뷰 조건부 |
| (선택) 대응점 | RoMa / LoFTR | 뷰 간 드리프트 초기 추정 |

> 회색지대: DUSt3R / MASt3R / VGGT는 생성모델이 아니라 재구성모델이라 금지 대상은
> 아니라고 봅니다. 다만 실사 원근 사진으로 학습돼서 직교 렌더에 약할 가능성이
> 큽니다. 넣을지는 결정 주세요. 기본값은 **제외**로 두겠습니다.

### 라이브러리

| 용도 | 패키지 |
|------|--------|
| 미분가능 래스터화 | `nvdiffrast` (주력), `pytorch3d` (대안) |
| 최적화 가능 메쉬 | `kaolin` (FlexiCubes 포함) |
| 메쉬 처리 | `open3d`, `pymeshlab`, `libigl`, `trimesh` |
| Normal 적분 | BiNI 공식 구현 |
| GT 렌더 / 최종 정리 | `bpy` (Blender as module) |
| 수치 | `torch`, `numpy`, `scipy`, `opencv` |

### 컴퓨트

이 컨테이너: GPU 없음, 4코어, 15GB RAM. 따라서:
- **CPU 단계** (여기서 실행): S2 정준화, S3 visual hull, S5 보정, S7 메쉬 정리,
  평가 지표, GT 렌더 (Blender CPU)
- **GPU 단계** (Kaggle 노트북, T4/P100): S1 분할, S4 normal/depth 추론, S6 공동 최적화

Kaggle 무료 GPU는 세션당 제한이 있으니 S6는 체크포인트 저장 필수. 형상 파라미터만
저장하면 되니 용량은 작습니다.

---

## 6. 마일스톤

| # | 산출물 | 성공 기준 |
|---|--------|-----------|
| M0 | GT 테스트 하네스 | 알려진 메쉬 → 6뷰 렌더 → Chamfer/IoU 계산 코드가 돎 |
| M1 | 실루엣 → visual hull | GT 대비 IoU > 0.75. **여기서 이미 메쉬가 나옵니다.** 뚱뚱하지만 절대 안 터짐 |
| M2 | normal/depth 추정 + affine 보정 | 뷰별 깊이가 하나의 좌표계에 정렬됨 (보정 전후 정렬 오차 그래프) |
| M3 | 공동 최적화 (GT 입력) | Chamfer가 M1 대비 유의미하게 감소. **이걸 통과해야 생성 이미지 투입** |
| M4 | 생성 이미지 투입 | 메쉬가 안 터짐. 드리프트 파라미터 크기 리포트 |
| M5 | 디테일 베이킹 | 하이폴리 + normal/displacement map |
| M6 | 정리 | 리메쉬, UV, 익스포트 |

M1이 중요합니다. **1번 마일스톤부터 이미 "터지지 않는 메쉬"가 손에 있습니다.**
품질은 낮지만 하한선이 확보되고, 이후 모든 단계는 그 하한선에서 개선만 합니다.
지금까지의 접근에는 하한선이 없었어요 — 그래서 실패가 "조금 나쁨"이 아니라
"완전 폭발"이었던 겁니다.

---

## 7. 결정 필요 사항

1. 다리 모호성 → (a) 6뷰 고수 후 후처리 / (b) 3/4 보조뷰 추가. 기본값 (a).
2. DUSt3R/MASt3R/VGGT 허용 여부. 기본값 제외.
3. S6 표현 → FlexiCubes vs hash-grid neural SDF. 기본값 FlexiCubes (메쉬가 직접
   나오고 정규화가 쉬움).
4. 최종 폴리곤 예산 및 출력 포맷 (.obj / .glb / .fbx).
