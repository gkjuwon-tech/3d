"""
문장 -> 에너지 수식 번역기 실행기.

    python translate.py --model runs/v0/model.pt "철수는 영희보다 사과가 3개 많고 둘이 합쳐 17개다. 철수의 사과는 몇 개?"
    python translate.py --model runs/v0/model.pt          # 대화형
"""

from __future__ import annotations
import argparse
import sys

import torch

import elang as E
from model import ECLIP, Ear
from solver import announce, solve
from train import greedy_decode


def translate(text: str, model: ECLIP, ear: Ear) -> str:
    words = E.split_words(text)
    nums, _ = E.extract_numbers(words)
    feats = ear.encode([words])[0]

    class _S:  # greedy_decode 가 요구하는 최소 정보
        pass
    s = _S()
    s.words, s.nums, s.tokens = words, nums, ["BOS", "EOS"]
    toks = greedy_decode(model, [{"sample": s, **feats}])[0]

    lines = [f"📝 입력: {text}", f"🔢 원문 숫자: {nums}", "", "🔥 추론기에게 가는 에너지 수식 (사연 없음):"]
    try:
        prog = E.parse(toks)
    except E.ELangError as e:
        return "\n".join(lines + [f"  번역 실패: {e}", "  토큰: " + " ".join(toks)])
    body = E.render_program(prog, nums, words).split("--- 꼬리표 장부")
    lines += ["  " + l for l in body[0].strip().splitlines()]
    lines += ["", "📒 꼬리표 장부 (역번역 전용):"]
    for v, idxs in prog.labels.items():
        lines.append(f"  {v} = " + " ".join(words[i] for i in idxs if i < len(words)))
    lines.append("  답 구절 = " + " ".join(words[i] for i in prog.answer_label if i < len(words)))
    res = solve(prog, nums)
    lines += ["", f"👮 검수관: {res['status']}"]
    if res["solution"] is not None:
        lines.append(f"  E=0 해: { {k: round(v, 4) for k, v in res['solution'].items()} }")
        lines += ["", "🎤 " + announce(prog, res["solution"], words)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/v0/model.pt")
    ap.add_argument("text", nargs="*")
    args = ap.parse_args()
    torch.set_num_threads(4)
    model = ECLIP()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()
    ear = Ear()
    if args.text:
        print(translate(" ".join(args.text), model, ear))
        return
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(translate(line, model, ear))
            print("-" * 80)


if __name__ == "__main__":
    main()
