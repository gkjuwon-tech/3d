"""
E-CLIP 번역기 학습: 역방향 공장 데이터 -> 귀 캐시 -> (수식 쓰기 + CLIP 짝 맞추기) 동시 학습 -> 검문소 채점.

    python train.py --n-train 24000 --epochs 20 --out runs/v0
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F

import elang as E
import factory
from solver import solve
from model import ECLIP, Ear, MAX_TGT, formula_depths, formula_part, mutate_formula


# ==================================================================== 데이터
def build_split(n: int, seed: int, ood: bool, ear: Ear, batch: int = 64) -> List[Dict]:
    rng = random.Random(seed)
    samples = [factory.make_sample(rng, ood=ood) for _ in range(n)]
    out = []
    for i in range(0, n, batch):
        chunk = samples[i: i + batch]
        feats = ear.encode([s.words for s in chunk])
        for s, f in zip(chunk, feats):
            out.append({"sample": s, **f})
    return out


def load_or_build(path: str, n: int, seed: int, ood: bool, ear_holder: Dict) -> List[Dict]:
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    if "ear" not in ear_holder:
        ear_holder["ear"] = Ear()
    t0 = time.time()
    data = build_split(n, seed, ood, ear_holder["ear"])
    torch.save(data, path)
    print(f"  built {path}: {n} samples in {time.time() - t0:.0f}s", flush=True)
    return data


def pad(seqs: List[torch.Tensor], value) -> torch.Tensor:
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L) + tuple(seqs[0].shape[1:]), value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def collate(items: List[Dict], rng: random.Random) -> Dict[str, torch.Tensor]:
    h = pad([it["h"] for it in items], 0.0)
    word_id = pad([it["word_id"] for it in items], -1)
    num_slot = pad([it["num_slot"] for it in items], -1)
    mem_mask = pad([torch.ones(len(it["h"]), dtype=torch.bool) for it in items], False)
    tgt = pad([torch.tensor([E.TOK2ID[t] for t in it["sample"].tokens]) for it in items], E.PAD_ID)

    def fids(toks):
        return torch.tensor([E.TOK2ID[t] for t in toks]), torch.tensor(formula_depths(toks))

    pos_f, neg_f = [], []
    for it in items:
        s = it["sample"]
        fp = formula_part(s.tokens)
        pos_f.append(fids(fp))
        neg = mutate_formula(fp, len(s.nums), rng) or fp
        neg_f.append(fids(neg))
    f_all = pos_f + neg_f
    return {
        "h": h, "word_id": word_id, "num_slot": num_slot, "mem_mask": mem_mask, "tgt": tgt,
        "f_ids": pad([a for a, _ in f_all], E.PAD_ID), "f_depth": pad([b for _, b in f_all], 0),
    }


# ==================================================================== 디코딩 (문법 강제)
@torch.no_grad()
def greedy_decode(model: ECLIP, items: List[Dict]) -> List[List[str]]:
    model.eval()
    b = collate(items, random.Random(0))
    mem = model.memory(b["h"], b["word_id"], b["num_slot"], b["mem_mask"])
    wv, nv = model.pointer_tables(mem, b["word_id"], b["num_slot"])
    B = len(items)
    states = [E.GrammarState(len(it["sample"].nums), len(it["sample"].words)) for it in items]
    seqs = [[E.BOS_ID] for _ in range(B)]
    done = [False] * B
    for _ in range(MAX_TGT - 1):
        tgt_in = pad([torch.tensor(s) for s in seqs], E.PAD_ID)
        logits = model.decode_logits(tgt_in, mem, b["mem_mask"], wv, nv)
        lengths = [len(s) for s in seqs]
        for i in range(B):
            if done[i]:
                continue
            row = logits[i, lengths[i] - 1]
            allowed = states[i].allowed_ids()
            masked = torch.full_like(row, -1e9)
            masked[allowed] = row[allowed]
            nxt = int(masked.argmax())
            seqs[i].append(nxt)
            states[i].step(E.VOCAB[nxt])
            if nxt == E.EOS_ID:
                done[i] = True
        if all(done):
            break
    return [[E.VOCAB[t] for t in s] for s in seqs]


def evaluate(model: ECLIP, data: List[Dict], bs: int = 64) -> Dict:
    agg = defaultdict(float)
    by_gen = defaultdict(lambda: [0, 0])
    examples = []
    for i in range(0, len(data), bs):
        chunk = data[i: i + bs]
        outs = greedy_decode(model, chunk)
        for it, toks in zip(chunk, outs):
            s = it["sample"]
            chk = E.checkpoints(toks, s.nums, len(s.words), s.truth, s.read_var, s.words)
            ok_all = all(chk.values())
            for k, v in chk.items():
                agg[k] += v
            agg["all_pass"] += ok_all
            agg["exact_formula"] += formula_part(toks) == formula_part(s.tokens)
            # 진짜 번역 정확도: 검수관이 번역된 수식을 직접 풀었을 때 유일한 해가 정답과 같은가
            solved_ok = False
            try:
                prog = E.parse(toks)
                r = solve(prog, s.nums)
                if r["status"] == "unique" and r["solution"] is not None:
                    gold = s.truth[s.read_var]
                    solved_ok = abs(r["solution"][prog.read[0]] - gold) <= 1e-6 * max(1.0, abs(gold))
            except Exception:
                pass
            agg["solved_correct"] += solved_ok
            agg["exact_full"] += toks == s.tokens
            by_gen[s.gen][0] += solved_ok
            by_gen[s.gen][1] += 1
            if len(examples) < 6 and not solved_ok:
                examples.append({"text": s.text, "pred": " ".join(toks), "gold": " ".join(s.tokens)})
    n = len(data)
    res = {k: round(100.0 * v / n, 2) for k, v in agg.items()}
    res["by_gen_all_pass"] = {g: round(100.0 * a / t, 1) for g, (a, t) in sorted(by_gen.items())}
    res["fail_examples"] = examples
    return res


# ==================================================================== 학습
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v0")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--n-eval", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip-weight", type=float, default=0.3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help="빠른 실험용: 학습 데이터 앞 N개만 사용")
    ap.add_argument("--resume", action="store_true", help="out/ckpt.pt (또는 out/model.pt) 에서 이어서 학습")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.data, exist_ok=True)

    ear_holder: Dict = {}
    print("[1/3] 공장 가동 + 귀 캐시", flush=True)
    train = load_or_build(f"{args.data}/train_{args.n_train}.pt", args.n_train, 1, False, ear_holder)
    val = load_or_build(f"{args.data}/val_{args.n_eval}.pt", args.n_eval, 2, False, ear_holder)
    ood = load_or_build(f"{args.data}/ood_{args.n_eval}.pt", args.n_eval, 3, True, ear_holder)
    ear_holder.clear()
    if args.subset:
        train = train[: args.subset]

    model = ECLIP()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2/3] 모델 파라미터 {n_params / 1e6:.1f}M (귀 117M 은 얼림)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train) / args.bs)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    rng = random.Random(args.seed)

    log = []
    start_ep = 1
    if args.resume:
        if os.path.exists(f"{args.out}/ckpt.pt"):
            ck = torch.load(f"{args.out}/ckpt.pt", weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_ep, log = ck["epoch"] + 1, ck["log"]
        else:
            # 가중치만 남은 경우: 스케줄러를 해당 지점까지 빨리 감기 (옵티마이저 모멘텀은 새로 시작)
            model.load_state_dict(torch.load(f"{args.out}/model.pt"))
            start_ep = args.resume_epoch + 1 if hasattr(args, "resume_epoch") else 2
            for _ in range((start_ep - 1) * steps_per_epoch):
                sched.step()
        print(f"  이어서 학습: epoch {start_ep}부터", flush=True)
    print("[3/3] 학습 시작", flush=True)
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        idx = list(range(len(train)))
        rng.shuffle(idx)
        t0 = time.time()
        tot_ce = tot_clip = 0.0
        for st in range(steps_per_epoch):
            items = [train[j] for j in idx[st * args.bs: (st + 1) * args.bs]]
            b = collate(items, rng)
            mem = model.memory(b["h"], b["word_id"], b["num_slot"], b["mem_mask"])
            wv, nv = model.pointer_tables(mem, b["word_id"], b["num_slot"])
            logits = model.decode_logits(b["tgt"][:, :-1], mem, b["mem_mask"], wv, nv)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), b["tgt"][:, 1:].reshape(-1),
                                 ignore_index=E.PAD_ID, label_smoothing=0.05)
            # CLIP: 문장 B개 vs 수식 2B개 (정답 B + 한 끗 차이 가짜 B)
            te = model.text_embed(mem, b["mem_mask"])
            fe = model.formula_embed(b["f_ids"], b["f_depth"])
            scale = model.logit_scale.exp().clamp(max=100)
            sim = scale * te @ fe.t()
            labels = torch.arange(len(items))
            clip = 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim[:, : len(items)].t(), labels))
            loss = ce + args.clip_weight * clip
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_ce += ce.item()
            tot_clip += clip.item()
        msg = {"epoch": ep, "ce": round(tot_ce / steps_per_epoch, 4), "clip": round(tot_clip / steps_per_epoch, 4),
               "sec": round(time.time() - t0, 1)}
        msg["val"] = {k: v for k, v in evaluate(model, val[:300]).items() if k not in ("fail_examples", "by_gen_all_pass")}
        print(json.dumps(msg, ensure_ascii=False), flush=True)
        log.append(msg)
        torch.save(model.state_dict(), f"{args.out}/model.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "log": log}, f"{args.out}/ckpt.pt")

    final = {"val": evaluate(model, val), "ood": evaluate(model, ood)}
    with open(f"{args.out}/report.json", "w", encoding="utf-8") as f:
        json.dump({"log": log, "final": final, "args": vars(args)}, f, indent=2, ensure_ascii=False)
    for split in ("val", "ood"):
        r = final[split]
        print(f"\n=== {split} ===")
        print({k: v for k, v in r.items() if k not in ("fail_examples", "by_gen_all_pass")})
        print("문제 유형별 검수관 정답률:", r["by_gen_all_pass"])
        for ex in r["fail_examples"][:3]:
            print("  ✗", ex["text"], "\n    pred:", ex["pred"], "\n    gold:", ex["gold"])


if __name__ == "__main__":
    main()
