#!/usr/bin/env python3
"""Generate one image through Codex CLI signed in with a ChatGPT plan.

Codex's built-in image generation (the `$imagegen` skill, gpt-image) is
available to ChatGPT-plan logins and bills the plan's Codex allowance -- the
officially supported way to use a ChatGPT subscription from a script. No API
key, and nothing here calls ChatGPT's backend directly.

One-time setup per machine (a cloud container forgets it when reclaimed):
    npm install -g @openai/codex
    codex login --device-auth   # enable device-code sign-in in ChatGPT
                                # Settings > Security first

Run:
    python3 tools/gen_image.py --prompt "..." --out path.png [--ref a.png ...]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time


def generate(prompt, out, refs=(), timeout=600):
    out = os.path.abspath(out)
    work = tempfile.mkdtemp(prefix="gen_image_")
    name = "result.png"
    ask = (f"Use your image generation tool to create exactly ONE image.\n\n"
           f"{prompt}\n\n"
           + ("The attached image(s) are references; follow them as the prompt "
              "says.\n\n" if refs else "")
           + f"Save the generated image into the current working directory as "
             f"{name}. Do nothing else: no other files, no edits, no commands "
             f"beyond saving the image.")
    cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral",
           "-s", "workspace-write", "-C", work]
    for r in refs:
        cmd += ["-i", os.path.abspath(r)]
    # the prompt goes on stdin: -i takes any number of files and would read
    # a trailing prompt argument as one more
    t0 = time.time()
    r = subprocess.run(cmd, input=ask, capture_output=True, text=True,
                       timeout=timeout)
    got = os.path.join(work, name)
    if r.returncode or not os.path.exists(got):
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        shutil.rmtree(work, ignore_errors=True)
        raise RuntimeError(f"codex exec produced no image (exit {r.returncode})")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    shutil.move(got, out)
    shutil.rmtree(work, ignore_errors=True)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", action="append", default=[],
                    help="reference image to attach (repeatable)")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    if not shutil.which("codex"):
        sys.exit("codex not found: npm install -g @openai/codex")
    st = subprocess.run(["codex", "login", "status"], capture_output=True, text=True)
    if "ChatGPT" not in (st.stdout + st.stderr):
        sys.exit("not signed in with ChatGPT: codex login --device-auth")
    dt = generate(a.prompt, a.out, a.ref, a.timeout)
    print(f"wrote {a.out}  ({dt:.0f}s)")


if __name__ == "__main__":
    main()
