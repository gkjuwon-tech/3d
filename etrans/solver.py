"""
검수관: 번역된 에너지 수식이 정말 E = 0 인 해를 갖는지 일반 도구(SymPy, 정수 탐색)로 확인합니다.
나중에 만들 추론기의 대역이 아니라, 번역 품질을 채점하고 왕복 데모를 보여주기 위한 도구입니다.
"""

from __future__ import annotations
import itertools
import math
from typing import Dict, List, Optional, Sequence

import sympy as sp

import elang as E


def _to_sympy(e: E.Expr, syms: Dict[str, sp.Symbol], nums: Sequence[float]):
    if isinstance(e, str):
        if e in syms:
            return syms[e]
        if e in E.CONSTS:
            return sp.nsimplify(E.CONSTS[e])
        if e in E.NUMS:
            return sp.nsimplify(nums[int(e[1:])])
        raise E.ELangError(e)
    op = e[0]
    a = _to_sympy(e[1], syms, nums)
    if op == "NEG":
        return -a
    if op == "SQ":
        return a ** 2
    if op == "ABS":
        return sp.Abs(a)
    if op == "RSQ":
        return sp.Max(a, 0) ** 2
    b = _to_sympy(e[2], syms, nums)
    return {"ADD": a + b, "SUB": a - b, "MUL": a * b, "DIV": a / b}[op]


def solve(prog: E.Program, nums: Sequence[float], int_range: int = 2000) -> Dict:
    """E = 0 인 해를 찾습니다. 반환: {"status", "solution", "energy", "n_solutions"}"""
    names = [f"v{i}" for i in range(len(prog.var_types))]
    syms = {n: sp.Symbol(n, real=True) for n in names}
    eqs = []
    has_ineq = False
    for t in prog.terms:
        if t[0] == "SQ" or t[0] == "ABS":
            # 뿌리가 제곱/절댓값인 항은 "안쪽 식 = 0" 이라는 등식
            eqs.append(_to_sympy(t[1], syms, nums))
        else:
            has_ineq = True

    candidates: List[Dict[str, float]] = []
    if eqs and not has_ineq:
        try:
            sols = sp.solve(eqs, [syms[n] for n in names], dict=True)
        except Exception:
            sols = []
        for s in sols:
            if all(syms[n] in s for n in names) and all(s[syms[n]].is_real for n in names):
                candidates.append({n: float(s[syms[n]]) for n in names})
            elif s:
                return {"status": "underdetermined", "solution": None, "energy": None, "n_solutions": math.inf}
    elif len(names) <= 2:
        # 부등식이 섞이면 정수 격자 탐색 (작은 문제 전용)
        rng = range(0, int_range + 1) if len(names) == 1 else range(0, 201)
        for combo in itertools.product(rng, repeat=len(names)):
            env = dict(zip(names, map(float, combo)))
            if E.energy(prog, env, nums) <= 1e-9:
                candidates.append(env)
                if len(candidates) >= 2:
                    break
    # 정수형 변수는 정수 해만 인정
    good = []
    for c in candidates:
        if all(prog.var_types[i] != "INT" or abs(c[n] - round(c[n])) < 1e-9 for i, n in enumerate(names)):
            if E.energy(prog, c, nums) <= 1e-6 * E.term_scale(prog, nums):
                good.append(c)
    if not good:
        return {"status": "no_solution", "solution": None, "energy": None, "n_solutions": 0}
    best = good[0]
    return {"status": "unique" if len(good) == 1 else "multiple", "solution": best,
            "energy": E.energy(prog, best, nums), "n_solutions": len(good)}


def announce(prog: E.Program, sol: Dict[str, float], words: Sequence[str]) -> str:
    """템플릿 아나운서: 장부를 보고 답을 말로 바꿉니다 (지어내는 말 없음)."""
    var = prog.read[0]
    val = sol[var]
    sval = str(int(round(val))) if abs(val - round(val)) < 1e-9 else f"{val:.4g}"
    label = " ".join(words[i] for i in prog.labels.get(var, []) if i < len(words)).strip()
    if not label:
        return f"사용자님 문제의 답은 {sval}입니다."
    if label.endswith(("는", "은", "가", "이")):
        return f"사용자님 문제의 답: {label} {sval}입니다."
    return f"사용자님 문제의 답: {label} → {sval}"
