# Cinder Basilisk — 6뷰 직교 레퍼런스 프롬프트 팩

Gemini 앱(구독)에서 **손으로 돌리는** 버전. API 쿼터 없이 진행 가능.

---

## 0. 먼저 알아야 할 5개 규칙

이거 안 지키면 아래 프롬프트 아무리 좋아도 디자인 튑니다.

1. **채팅 1개에서 6뷰 다 뽑기.** 새 채팅 열면 컨텍스트 날아가고 즉시 다른 생물이 나옴.
2. **매 턴마다 이전 이미지를 명시적으로 지목.** 대화에 남아있어도 "그 이미지 보고 해"라고
   말 안 하면 텍스트만 다시 해석해서 새로 디자인함. 첨부 가능하면 첨부까지.
3. **왼쪽 측면은 생성하지 말고 오른쪽 측면을 좌우 반전.** 이 생물은 완전 좌우대칭이라
   미러가 정답이고, 생성하면 100% 미세하게 달라짐. 1뷰 버는 게 아니라 **일관성을
   공짜로 얻는** 거예요. (검증용으로 굳이 생성하고 싶으면 §4에 프롬프트 있음)
4. **숫자가 제일 먼저 무너짐.** 뿔 7개 → 6개 → 9개. 매 뷰마다 §5 체크리스트로 세세요.
   틀리면 그 뷰는 버리고 재생성. 통과한 것만 다음 뷰의 레퍼런스로 올립니다. 이게
   "레퍼런스 체인"의 핵심 — **오염된 프레임을 체인에 넣지 않는 것.**
5. **생성 순서는 정보량 순.** 측면(길이 비율) → 정면(폭) → 위(팔다리 배치) →
   뒤 → 아래. 측면이 마스터입니다.

---

## 1. 생성 순서와 체인 구조

| 턴 | 뷰 | 첨부할 레퍼런스 | 파일명 |
|----|-----|----------------|--------|
| 1 | **오른쪽 측면** (마스터) | 없음 | `02_right.png` |
| 2 | 정면 | 측면 | `01_front.png` |
| 3 | 위 | 측면 + 정면 | `05_top.png` |
| 4 | 뒤 | 정면 + 측면 + 위 | `03_back.png` |
| 5 | 아래 | 측면 + 정면 + 위 | `06_bottom.png` |
| — | 왼쪽 측면 | `02_right.png` 좌우반전 | `04_left.png` |

레퍼런스는 4장 이하로 유지하세요. 더 넣으면 모델이 뷰들을 **평균내기** 시작해서
각 뷰가 애매한 3/4 앵글로 수렴합니다.

---

## 2. 디자인 바이블 (매 턴 그대로 복붙)

아래 블록은 **한 글자도 바꾸지 말고** 6번 다 똑같이 붙이세요. 숫자가 고정값입니다.

```
CREATURE: "Cinder Basilisk" — original design. A hexapodal volcanic apex
predator, roughly 4 meters long, built like a cross between a deep-sea
crustacean and a wingless dragon.

LOCKED ANATOMY — these counts are exact and must be identical in every view:
- Head: elongated armored skull, narrow wedge snout, split lower mandible with
  4 inward-curving fangs. Mouth CLOSED.
- Eyes: 4 per side (8 total) in a descending arc — 2 large forward, 2 small
  below — glowing pale amber.
- Crown: a fan of exactly 7 backward-swept bone spines. Tallest in the center,
  shortest at the outer edges, mirrored left/right.
- Neck: exactly 6 overlapping obsidian carapace plates, each rimmed with a
  thin cooling-magma seam.
- Torso: broad segmented chest shield, 10 raised rib ridges, one glowing
  orange fissure running down the exact center of the spine.
- Limbs: exactly 6 legs, 3 per side.
    front pair  — heaviest, 4-clawed hands
    middle pair — shortest
    rear pair   — digitigrade with hooked heel spurs
  Every foot: 3 forward claws + 1 rear claw.
- Tail: long tapering counterweight tail, exactly 12 armor rings, ending in a
  cluster of exactly 5 jagged basalt crystals.
- Surface: matte black basalt plating, fine hexagonal micro-fracture texture,
  molten orange light bleeding from the seam between every plate.
- Perfectly bilaterally symmetric. No asymmetric details anywhere.

POSE — rigid neutral reference stance, identical in every view:
Standing flat and level on all 6 legs. Head straight forward and level, neck
straight, tail extended straight back and horizontal, limbs straight and
evenly spaced. NOT moving, NOT roaring, NOT dramatically posed.

RENDER SPEC — this is a MODELING REFERENCE SHEET, not concept art:
- Strict ORTHOGRAPHIC projection. Zero perspective, zero lens distortion,
  zero foreshortening. Parallel lines stay parallel.
- Flat neutral studio lighting, even from all directions. No rim light, no
  cast shadows, no ground plane, no fog, no atmosphere, no bloom.
- Plain flat neutral mid-grey background (#808080), completely empty.
- Square 1:1 canvas. Creature centered, fully inside frame with a small even
  margin. Nothing cropped.
- High detail, high polygon count, crisp readable silhouette and panel lines.
- NO text, NO labels, NO watermark, NO grid, NO turnaround strip.
  EXACTLY ONE single view fills the entire image.
```

---

## 3. 턴별 프롬프트

각 턴은 **[디자인 바이블 전체] + 아래 해당 블록**입니다. 자기완결형으로 붙이세요
(대화 기억에 의존하면 드리프트 납니다).

### 턴 1 — 오른쪽 측면 (마스터)

```
VIEW REQUIRED — RIGHT SIDE view. Camera at exactly 90 degrees to the
creature's right flank, at the creature's mid-height. Full profile: snout near
one edge of the frame, tail crystals near the other, body horizontal. All
three right-side legs visible and distinguishable. This image will be the
master reference for all other views, so make the proportions, plate layout
and every locked count above unambiguous and clearly readable.
```

### 턴 2 — 정면

```
The attached image is the canonical RIGHT SIDE view of this creature.

This is the SAME physical object photographed from a different camera angle.
Do not redesign it. Do not reinterpret it. Change ONLY the camera.

VIEW REQUIRED — FRONT view. Camera directly in front, at mid-height, looking
straight down the creature's centerline toward the tail. The head faces the
camera dead-on. The left and right halves of the image must be exact mirror
images of each other. All 6 legs visible: the front pair widest, middle and
rear pairs behind them.

Match the reference exactly: same body height in the frame, same vertical
centering, same colour, same plate count, same scale. Head height, shoulder
height and hip height must line up with the side view if the two images were
stacked.
```

### 턴 3 — 위

```
The attached images are the canonical RIGHT SIDE and FRONT views of this
creature. Same physical object, new camera only. Do not redesign anything.

VIEW REQUIRED — TOP view. A true plan view from directly overhead looking
straight down. Visible: the back plating, the glowing centerline spine
fissure, the crown spine fan from above, and the full spread of all 6 legs.
The spine runs vertically through the exact center of the image, snout at the
top edge, tail crystals at the bottom edge. Perfectly mirrored left to right.

The creature's total nose-to-tail length must match the side view's length
exactly, at the same scale.
```

### 턴 4 — 뒤

```
The attached images are the canonical FRONT, RIGHT SIDE and TOP views of this
creature. Same physical object, new camera only. Do not redesign anything.

VIEW REQUIRED — BACK view. Camera directly behind, at mid-height, looking up
the centerline toward the head. The tail points straight at the camera, so the
tail crystal cluster is seen end-on near the center of the frame. The 7 crown
spines are seen from behind, rising above the back. The rear pair of legs is
nearest, with the middle and front pairs visible beyond them. The left and
right halves must be exact mirror images.

Match the front view's body height and centering exactly.
```

### 턴 5 — 아래

```
The attached images are the canonical RIGHT SIDE, FRONT and TOP views of this
creature. Same physical object, new camera only. Do not redesign anything.

VIEW REQUIRED — BOTTOM view. A true underside plan view from directly
underneath looking straight up. Visible: the belly plating, the underside of
the jaw and throat, and the soles of all 6 feet with their claws
(3 forward + 1 rear on each foot). Orientation must be the exact vertical
mirror of the top view: snout at the top edge, tail crystals at the bottom
edge, perfectly mirrored left to right.

Same total length and scale as the top view.
```

### 왼쪽 측면 — 생성 안 함

`02_right.png`를 좌우반전 → `04_left.png`. 끝.
macOS 미리보기: 도구 ▸ 좌우 반전. Windows 사진: 편집 ▸ 회전/반전.
ImageMagick: `magick 02_right.png -flop 04_left.png`

---

## 4. (선택) 왼쪽 측면을 굳이 생성해서 검증하고 싶다면

```
The attached image is the canonical RIGHT SIDE view of this creature.

VIEW REQUIRED — LEFT SIDE view. This must be the precise horizontal mirror of
the attached right side view: the snout points the opposite direction, and
every proportion, plate seam, spine, rib ridge, leg position and tail ring is
identical. Nothing new, nothing removed, nothing shifted. Same scale, same
vertical centering.
```

미러본과 비교해서 차이 나면 **미러본을 쓰세요.**

---

## 5. 뷰별 합격 체크리스트

레퍼런스 체인에 올리기 전에 매번 세세요. 하나라도 틀리면 재생성.

| 항목 | 정답 |
|------|------|
| 왕관 뿔 | **7개** (중앙 최고, 바깥 최저) |
| 목 판 | **6장** |
| 갈비 융기 | **10개** |
| 다리 | **6개** (편측 3개) |
| 발톱 | 발마다 앞 **3** + 뒤 **1** |
| 눈 | 편측 **4개** (큰 거 2 + 작은 거 2) |
| 송곳니 | **4개**, 입은 닫힘 |
| 꼬리 고리 | **12개** |
| 꼬리 끝 결정 | **5개** |
| 등 균열 | 정중선 **1줄**, 주황 발광 |
| 배경 | 균일한 회색, 완전히 빔 |
| 투시 | 원근 없음 — 판 선들이 평행 |
| 그림자 | 바닥 그림자 없음 |
| 프레임 | 정사각, 안 잘림, 중앙 정렬 |
| 뷰 개수 | 이미지당 **1뷰** |

**검증 꼼수:** 이미지 다시 첨부하고 Gemini에 이렇게 물어보세요 —
`Count the crown spines, the legs, the tail armor rings and the tail crystals
in this image. Answer with numbers only.`
자기가 만든 거 자기가 셀 때 은근 정직합니다.

---

## 6. 증상별 수리 프롬프트

| 증상 | 대응 |
|------|------|
| 원근이 슬금슬금 들어옴 | `Regenerate with strict orthographic projection. This is a technical blueprint elevation, not a photograph. Parallel lines must stay parallel. No vanishing point.` |
| 생물이 커졌다 작아짐 | `The creature must occupy the same height in the frame as the attached reference — same scale, same centering. Rescale, do not redesign.` |
| 개수가 틀림 | `You drew N spines. The design has exactly 7. Fix only the spine count. Change nothing else.` (숫자 직접 지적할수록 잘 먹음) |
| 멋있게 포즈 잡음 | `Return to the rigid neutral reference stance: level head, straight neck, horizontal tail, all six legs straight and evenly spaced. This is a T-pose equivalent, not an action shot.` |
| 다른 생물이 됨 | 그 턴 버리고, 마스터 측면 하나만 첨부해서 다시. 체인이 오염됨. |
| 6뷰 시트를 한 장에 그림 | `ONE single view only. Do not produce a turnaround sheet or a grid of views.` |
| 배경에 안개/바닥 깜 | `Flat empty #808080 background. No ground plane, no shadow, no atmosphere, no gradient.` |

---

## 7. 뽑은 거 넘기는 방법

```
refs/cinder_basilisk/
  01_front.png
  02_right.png
  03_back.png
  04_left.png     ← 02를 좌우반전
  05_top.png
  06_bottom.png
```

이 파일명 그대로 주시면 제가 정렬 검증(뷰 간 높이/길이 매칭), 컨택트 시트,
그리고 다음 단계로 바로 넘어갑니다. 2K 이상, PNG로.
