"""
E-Lang: 자연어 문제를 '사연 없는' 에너지 수식으로 적는 공용 언어.

모델이 쓰는 형태는 전위(prefix) 토큰열입니다. 연산자마다 받는 인자 수가 정해져 있어서
"아직 몇 칸을 더 채워야 하는지"만 세면 문법을 100% 강제할 수 있습니다.

    VAR v0 INT  VAR v1 INT
    TERM SQ SUB SUB v0 v1 N0          # (v0 - v1 - n0)^2
    TERM SQ SUB ADD v0 v1 N1          # (v0 + v1 - n1)^2
    READ v0
    LBL v0 W0 W5  LBL v1 W2           # 꼬리표 장부: 원문 단어를 가리키는 포인터
    EOS

- 수식 부분(VAR/TERM/READ)에는 원문 단어가 들어갈 자리가 없습니다. 숫자도 N_k(원문 k번째 숫자)로만
  가리키므로 사연은 구조적으로 새어 나갈 수 없습니다.
- TERM 의 뿌리는 SQ / RSQ(relu 제곱) / ABS 중 하나라서 모든 항이 0 이상이고,
  모든 조건이 지켜질 때 정확히 E = 0 이 됩니다.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

MAX_VARS = 8
MAX_NUMS = 16
MAX_WORDS = 64
MAX_TERM_LEN = 40
MAX_LABEL_WORDS = 6

SPECIALS = ["PAD", "BOS", "EOS"]
KEYWORDS = ["VAR", "TERM", "READ", "LBL"]
TYPES = ["INT", "REAL"]
VARS = [f"v{i}" for i in range(MAX_VARS)]
ROOT_OPS = ["SQ", "RSQ", "ABS"]
BINARY_OPS = ["ADD", "SUB", "MUL", "DIV"]
UNARY_OPS = ["NEG", "SQ", "RSQ", "ABS"]
CONSTS = {"C0": 0.0, "C1": 1.0, "C2": 2.0, "C0.5": 0.5, "C10": 10.0, "C60": 60.0, "C100": 100.0, "C1000": 1000.0}
NUMS = [f"N{i}" for i in range(MAX_NUMS)]
WORDS = [f"W{i}" for i in range(MAX_WORDS)]

VOCAB: List[str] = list(dict.fromkeys(
    SPECIALS + KEYWORDS + TYPES + VARS + ROOT_OPS + BINARY_OPS + UNARY_OPS + list(CONSTS) + NUMS + WORDS
))
TOK2ID: Dict[str, int] = {t: i for i, t in enumerate(VOCAB)}
PAD_ID, BOS_ID, EOS_ID = TOK2ID["PAD"], TOK2ID["BOS"], TOK2ID["EOS"]

ARITY = {**{op: 2 for op in BINARY_OPS}, **{op: 1 for op in UNARY_OPS}}

_NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


# ============================================================== 원문 전처리
def split_words(text: str) -> List[str]:
    return text.split()


def extract_numbers(words: Sequence[str]) -> Tuple[List[float], List[int]]:
    """단어마다 앞에서부터 숫자를 찾아 원문 숫자 목록과, 단어별 숫자 슬롯(-1 = 숫자 없음)을 돌려줍니다."""
    nums: List[float] = []
    slot_of_word: List[int] = []
    for w in words:
        m = _NUM_RE.search(w)
        if m and len(nums) < MAX_NUMS:
            nums.append(float(m.group(0).replace(",", "")))
            slot_of_word.append(len(nums) - 1)
        else:
            slot_of_word.append(-1)
    return nums, slot_of_word


# ============================================================== 구문 트리
Expr = Union[str, Tuple]  # 잎: 토큰 문자열, 내부 노드: (op, child...)


@dataclass
class Program:
    var_types: List[str] = field(default_factory=list)
    terms: List[Expr] = field(default_factory=list)
    read: List[str] = field(default_factory=list)
    labels: Dict[str, List[int]] = field(default_factory=dict)  # 변수 -> 원문 단어 인덱스들


class ELangError(ValueError):
    pass


def _parse_expr(toks: List[str], i: int) -> Tuple[Expr, int]:
    if i >= len(toks):
        raise ELangError("식이 중간에 끝났습니다")
    t = toks[i]
    if t in ARITY:
        children = []
        j = i + 1
        for _ in range(ARITY[t]):
            c, j = _parse_expr(toks, j)
            children.append(c)
        return (t, *children), j
    if t in VARS or t in NUMS or t in CONSTS:
        return t, i + 1
    raise ELangError(f"식 자리에 올 수 없는 토큰: {t}")


def parse(tokens: Sequence[str]) -> Program:
    toks = [t for t in tokens if t not in ("BOS", "PAD")]
    if toks and toks[-1] == "EOS":
        toks = toks[:-1]
    prog = Program()
    i = 0
    while i < len(toks):
        kw = toks[i]
        if kw == "VAR":
            name, typ = toks[i + 1], toks[i + 2]
            if name != f"v{len(prog.var_types)}" or typ not in TYPES:
                raise ELangError(f"잘못된 VAR 선언: {name} {typ}")
            prog.var_types.append(typ)
            i += 3
        elif kw == "TERM":
            if toks[i + 1] not in ROOT_OPS:
                raise ELangError("TERM 의 뿌리는 SQ/RSQ/ABS 여야 합니다")
            expr, i = _parse_expr(toks, i + 1)
            prog.terms.append(expr)
        elif kw == "READ":
            prog.read.append(toks[i + 1])
            i += 2
        elif kw == "LBL":
            name = toks[i + 1]
            j = i + 2
            idxs = []
            while j < len(toks) and toks[j] in WORDS:
                idxs.append(int(toks[j][1:]))
                j += 1
            prog.labels[name] = idxs
            i = j
        else:
            raise ELangError(f"문장 첫머리에 올 수 없는 토큰: {kw}")
    if not prog.var_types or not prog.terms or not prog.read:
        raise ELangError("VAR, TERM, READ 가 최소 하나씩 필요합니다")
    declared = set(VARS[: len(prog.var_types)])
    for e in prog.terms:
        for leaf in _leaves(e):
            if leaf in VARS and leaf not in declared:
                raise ELangError(f"선언 안 된 변수 {leaf}")
    return prog


def _leaves(e: Expr):
    if isinstance(e, str):
        yield e
    else:
        for c in e[1:]:
            yield from _leaves(c)


# ============================================================== 에너지 계산
def eval_expr(e: Expr, env: Dict[str, float], nums: Sequence[float]) -> float:
    if isinstance(e, str):
        if e in env:
            return env[e]
        if e in CONSTS:
            return CONSTS[e]
        if e in NUMS:
            k = int(e[1:])
            if k >= len(nums):
                raise ELangError(f"원문에 {k}번째 숫자가 없습니다")
            return nums[k]
        raise ELangError(f"값을 모르는 잎: {e}")
    op = e[0]
    a = eval_expr(e[1], env, nums)
    if op == "NEG":
        return -a
    if op == "SQ":
        return a * a
    if op == "RSQ":
        return max(a, 0.0) ** 2
    if op == "ABS":
        return abs(a)
    b = eval_expr(e[2], env, nums)
    if op == "ADD":
        return a + b
    if op == "SUB":
        return a - b
    if op == "MUL":
        return a * b
    if op == "DIV":
        return a / b if b != 0 else math.inf
    raise ELangError(f"모르는 연산자 {op}")


def energy(prog: Program, env: Dict[str, float], nums: Sequence[float]) -> float:
    total = 0.0
    for t in prog.terms:
        v = eval_expr(t, env, nums)
        if not math.isfinite(v):
            return math.inf
        total += v
    return total


def term_scale(prog: Program, nums: Sequence[float]) -> float:
    """에너지를 문제 크기에 맞춰 상대값으로 보기 위한 눈금 (숫자가 클수록 허용 오차도 큼)."""
    mags = [abs(x) for x in nums] or [1.0]
    return max(1.0, max(mags)) ** 2


# ============================================================== 사람이 읽는 표기
_INFIX = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}


def render_expr(e: Expr, nums: Optional[Sequence[float]] = None, names: Optional[Dict[str, str]] = None) -> str:
    if isinstance(e, str):
        if names and e in names:
            return names[e]
        if e in NUMS and nums is not None and int(e[1:]) < len(nums):
            v = nums[int(e[1:])]
            return str(int(v)) if v == int(v) else str(v)
        if e in CONSTS:
            v = CONSTS[e]
            return str(int(v)) if v == int(v) else str(v)
        return e
    op = e[0]
    if op in _INFIX:
        return f"({render_expr(e[1], nums, names)} {_INFIX[op]} {render_expr(e[2], nums, names)})"
    inner = render_expr(e[1], nums, names)
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    return {"NEG": f"-({inner})", "SQ": f"({inner})^2", "RSQ": f"relu({inner})^2", "ABS": f"|{inner}|"}[op]


def render_program(prog: Program, nums: Optional[Sequence[float]] = None,
                   words: Optional[Sequence[str]] = None) -> str:
    lines = [f"VAR v{i} : {t.lower()}" for i, t in enumerate(prog.var_types)]
    for k, t in enumerate(prog.terms):
        lines.append(f"TERM t{k} = {render_expr(t, nums)}")
    lines.append("E = " + " + ".join(f"t{k}" for k in range(len(prog.terms))))
    lines.append("READ " + ", ".join(prog.read))
    if prog.labels:
        lines.append("--- 꼬리표 장부 (추론기는 못 봄) ---")
        for v, idxs in prog.labels.items():
            label = " ".join(words[i] for i in idxs if words and i < len(words)) if words else str(idxs)
            lines.append(f"{v} = {label}")
    return "\n".join(lines)


# ============================================================== 문법 강제 디코딩
class GrammarState:
    """접두 토큰열을 따라가며 다음에 올 수 있는 토큰 집합을 알려주는 자동기계.

    순서: VAR+ -> TERM+ -> READ -> LBL(선언 순서대로 모든 변수) -> EOS
    """

    def __init__(self, n_nums: int, n_words: int):
        self.n_nums = min(n_nums, MAX_NUMS)
        self.n_words = min(n_words, MAX_WORDS)
        self.phase = "start"
        self.n_vars = 0
        self.n_terms = 0
        self.need = 0       # 현재 TERM 식에서 아직 채워야 할 칸 수
        self.term_len = 0
        self.n_labeled = 0
        self.label_len = 0

    def _leaf_tokens(self) -> List[str]:
        return VARS[: self.n_vars] + NUMS[: self.n_nums] + list(CONSTS)

    def allowed(self) -> List[str]:
        p = self.phase
        if p == "done":
            return ["EOS"]
        if p == "start":
            return ["VAR"]
        if p == "var_name":
            return [f"v{self.n_vars}"]
        if p == "var_type":
            return TYPES
        if p == "stmt":
            opts = []
            if self.n_terms == 0 and self.n_vars < MAX_VARS:
                opts.append("VAR")
            opts.append("TERM")
            if self.n_terms >= 1:
                opts.append("READ")
            return opts
        if p == "term_root":
            return ROOT_OPS
        if p == "term":
            # 남은 칸을 잎으로 다 채울 수 있을 만큼만 연산자를 허용
            if MAX_TERM_LEN - self.term_len <= self.need + 1:
                return self._leaf_tokens()
            return list(dict.fromkeys(BINARY_OPS + UNARY_OPS)) + self._leaf_tokens()
        if p == "read":
            return VARS[: self.n_vars]
        if p == "lbl_kw":
            return ["LBL"]
        if p == "lbl_name":
            return [f"v{self.n_labeled}"]
        if p == "lbl_words":
            opts = WORDS[: self.n_words] if self.label_len < MAX_LABEL_WORDS else []
            if self.label_len >= 1:
                opts = opts + (["LBL"] if self.n_labeled < self.n_vars else ["EOS"])
            return opts
        raise RuntimeError(p)

    def allowed_ids(self) -> List[int]:
        return [TOK2ID[t] for t in self.allowed()]

    def step(self, tok: str) -> None:
        p = self.phase
        if p == "start":
            self.phase = "var_name"
        elif p == "var_name":
            self.phase = "var_type"
        elif p == "var_type":
            self.n_vars += 1
            self.phase = "stmt"
        elif p == "stmt":
            self.phase = {"VAR": "var_name", "TERM": "term_root", "READ": "read"}[tok]
        elif p == "term_root":
            self.need, self.term_len, self.phase = 1, 1, "term"
        elif p == "term":
            self.term_len += 1
            self.need += ARITY.get(tok, 0) - 1
            if self.need == 0:
                self.n_terms += 1
                self.phase = "stmt"
        elif p == "read":
            self.phase = "lbl_kw"
        elif p == "lbl_kw":
            self.phase = "lbl_name"
        elif p == "lbl_name":
            self.n_labeled += 1
            self.label_len = 0
            self.phase = "lbl_words"
        elif p == "lbl_words":
            if tok == "LBL":
                self.phase = "lbl_name"
            elif tok == "EOS":
                self.phase = "done"
            else:
                self.label_len += 1


def check_grammar(tokens: Sequence[str], n_nums: int, n_words: int) -> bool:
    g = GrammarState(n_nums, n_words)
    for t in tokens:
        if t in ("BOS", "PAD"):
            continue
        if t not in g.allowed():
            return False
        g.step(t)
        if t == "EOS":
            return True
    return False


# ============================================================== 4중 검문소 (정답을 아는 학습 데이터용)
def checkpoints(tokens: Sequence[str], nums: Sequence[float], n_words: int,
                truth: Dict[str, float], read_var: str, words: Sequence[str]) -> Dict[str, bool]:
    out = {"grammar": False, "answer_plug": False, "wrong_plug": False, "no_leak": False}
    out["grammar"] = check_grammar(tokens, len(nums), n_words)
    try:
        prog = parse(tokens)
    except (ELangError, IndexError):
        return out
    scale = term_scale(prog, nums)
    try:
        # 정답에 없는 변수를 지어낸 번역은 정답 대입에서 실패로 처리
        e_true = energy(prog, truth, nums)
        out["answer_plug"] = e_true <= 1e-9 * scale and prog.read == [read_var]
        wrong = dict(truth)
        wrong[read_var] = truth[read_var] + max(1.0, abs(truth[read_var]) * 0.1)
        out["wrong_plug"] = energy(prog, wrong, nums) > 1e-6
    except ELangError:
        pass
    formula_part = []
    for t in tokens:
        if t == "LBL":
            break
        formula_part.append(t)
    out["no_leak"] = not any(t in WORDS for t in formula_part)
    return out
