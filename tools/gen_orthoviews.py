#!/usr/bin/env python3
"""Generate a consistent 6-view orthographic reference sheet of a creature
using Gemini image generation with a reference chain.

Each view is generated with every previously accepted view passed back in as
an image reference, so the design locks instead of drifting.

Usage:
  GEMINI_API_KEY=... python3 tools/gen_orthoviews.py --out refs/cinder_basilisk
"""
import argparse, base64, json, os, sys, time, urllib.request, urllib.error

MODEL = "gemini-3-pro-image"
ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/"
            f"models/{MODEL}:generateContent")

# ---------------------------------------------------------------- the creature
SUBJECT = """\
A original creature design called the "Cinder Basilisk": a hexapodal
volcanic apex predator, roughly 4 meters long, built like a cross between a
deep-sea crustacean and a wingless dragon.

ANATOMY (must stay identical in every view):
- Head: elongated armored skull, narrow wedge snout, split lower mandible with
  four inward-curving fangs. FOUR eyes total per side arranged in a descending
  arc (2 large forward, 2 small below), glowing pale amber.
- Crown: a fan of seven backward-swept bone spines, tallest in the center,
  shortest at the outer edges, perfectly mirrored left/right.
- Neck: six overlapping obsidian carapace plates, each rimmed with a thin
  cooling-magma seam.
- Torso: broad segmented chest shield, ten ribs visible as raised ridges,
  glowing orange fissure running down the exact center of the spine.
- Limbs: SIX legs, three per side. Front pair heaviest with four-clawed
  hands, middle pair shortest, rear pair digitigrade with hooked heel spurs.
  All feet have three forward claws and one rear claw.
- Tail: long tapering counterweight tail, twelve armor rings, ending in a
  cluster of five jagged basalt crystals.
- Surface: matte black-basalt plating with fine hexagonal micro-fracture
  texture, molten orange light bleeding from the seams between every plate.
- The creature is perfectly bilaterally symmetric. No asymmetric details.

POSE: neutral rigid reference A-pose. Standing flat and level on all six legs,
head facing straight forward and level, neck straight, tail extended straight
back and horizontal, limbs straight and evenly spaced. It is NOT moving,
NOT roaring, NOT posed dramatically. Mouth closed."""

STYLE = """\
RENDER SPEC — this is a modeling reference sheet, not concept art:
- Strict ORTHOGRAPHIC projection. Zero perspective, zero lens distortion,
  zero foreshortening.
- Flat neutral studio lighting, even from all sides. No dramatic rim light,
  no cast shadows on the ground, no ground plane, no fog, no atmosphere.
- Plain flat neutral mid-grey background (#808080), completely empty.
- The full creature fits inside the frame with a small even margin. Nothing
  cropped. No motion blur, no depth of field, no glow bloom beyond the seams.
- High detail, high polygon, crisp readable silhouette and panel lines.
- NO text, NO labels, NO watermarks, NO grid, NO multiple views in one image,
  NO turnaround strip. Exactly ONE single view fills the image.
- The creature must occupy the SAME scale and be vertically/horizontally
  aligned identically to the reference images provided."""

VIEWS = [
    ("01_front",  "FRONT view. Viewed from directly in front, camera at the "
                  "creature's mid-height, looking straight down its centerline "
                  "toward the tail. The head faces the camera dead-on. Left and "
                  "right halves must be exact mirror images."),
    ("02_right",  "RIGHT SIDE view. Viewed from the creature's right side at "
                  "exact 90 degrees, camera at mid-height. Full profile: snout "
                  "at one edge, tail crystals at the other, body horizontal. "
                  "All three right-side legs visible."),
    ("03_back",   "BACK view. Viewed from directly behind, camera at mid-height, "
                  "looking up the centerline toward the head. The tail points at "
                  "the camera. The crown spines are seen from behind. Left and "
                  "right halves must be exact mirror images."),
    ("04_left",   "LEFT SIDE view. Viewed from the creature's left side at exact "
                  "90 degrees, camera at mid-height. This must be the precise "
                  "mirror of the right side view: snout pointing the opposite "
                  "direction, identical proportions and plate layout."),
    ("05_top",    "TOP view. Viewed from directly overhead looking straight down, "
                  "a true plan view. The back, spine fissure, crown spine fan and "
                  "the full spread of all six legs are visible. The spine runs "
                  "vertically through the center of the image. Perfectly mirrored "
                  "left to right."),
    ("06_bottom", "BOTTOM view. Viewed from directly underneath looking straight "
                  "up, a true underside plan view. The belly plating, the "
                  "underside of the jaw, and the soles of all six feet with their "
                  "claws are visible. Perfectly mirrored left to right."),
]


def post(payload, key, tries=5):
    req = urllib.request.Request(
        f"{ENDPOINT}?key={key}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:400]
            last = f"HTTP {e.code}: {body}"
            if e.code in (429, 500, 503) and i < tries - 1:
                wait = 2 ** (i + 1)
                print(f"    retry in {wait}s ({last[:90]})", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(last)
        except Exception as e:                       # network hiccup
            last = repr(e)
            if i < tries - 1:
                time.sleep(2 ** (i + 1)); continue
            raise RuntimeError(last)
    raise RuntimeError(last)


def extract_image(resp):
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            d = part.get("inlineData") or part.get("inline_data")
            if d and d.get("data"):
                return base64.b64decode(d["data"])
    raise RuntimeError("no image in response: " + json.dumps(resp)[:600])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="refs/cinder_basilisk")
    ap.add_argument("--size", default="2K", choices=["1K", "2K", "4K"])
    ap.add_argument("--only", default=None, help="regenerate one view by name")
    ap.add_argument("--max-refs", type=int, default=4,
                    help="how many prior views to feed back as references")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set")
    os.makedirs(args.out, exist_ok=True)

    chain = []   # (name, png bytes) accepted so far — the reference chain
    for name, view_spec in VIEWS:
        path = os.path.join(args.out, f"{name}.png")

        # reuse an existing view so --only can regenerate a single frame
        # while still feeding the rest of the chain
        if args.only and name != args.only and os.path.exists(path):
            chain.append((name, open(path, "rb").read()))
            print(f"[keep] {name}")
            continue

        parts = []
        refs = chain[-args.max_refs:]
        if refs:
            parts.append({"text":
                "REFERENCE IMAGES of this exact creature follow, in order: "
                + ", ".join(n.split("_", 1)[1].upper() + " view" for n, _ in refs)
                + ". Study them and reproduce the SAME creature with identical "
                  "anatomy, proportions, plate layout, spine count, limb count, "
                  "colour and scale. Do not redesign anything."})
            for n, b in refs:
                parts.append({"inlineData": {"mimeType": "image/png",
                                             "data": base64.b64encode(b).decode()}})

        parts.append({"text": f"{SUBJECT}\n\nVIEW REQUIRED — {view_spec}\n\n{STYLE}"})

        print(f"[gen ] {name}  (refs: {len(refs)})", flush=True)
        t0 = time.time()
        resp = post({
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["Image"],
                "imageConfig": {"aspectRatio": "1:1", "imageSize": args.size},
            },
        }, key)
        img = extract_image(resp)
        with open(path, "wb") as f:
            f.write(img)
        chain.append((name, img))
        print(f"       -> {path}  {len(img)//1024} KB  {time.time()-t0:.0f}s",
              flush=True)

    print("done:", args.out)


if __name__ == "__main__":
    main()
