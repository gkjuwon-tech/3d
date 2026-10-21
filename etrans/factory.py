"""
역방향 공장: 정답과 수식을 먼저 정해 놓고, 그걸 한국어 문제로 꾸며 냅니다.
정답을 처음부터 알고 있으니 검증 비용이 0이고, 무한히 찍어낼 수 있습니다.

템플릿 문법
- {#이름}   : 숫자 자리. 공장이 값을 채우고, 수식에서는 원문 k번째 숫자(N_k)로 바뀝니다.
- [[X:...]] : 변수 X 의 꼬리표(장부 이름표)로 쓸 원문 구절.
- 수식은 전위 토큰열이며 변수는 대문자 기호(X, A, B...), 숫자는 #이름 으로 적습니다.

같은 수식을 사과·구슬·마나포션 등 서로 다른 사연으로 꾸미기 때문에,
모델은 "사연이 달라도 수식은 같다"는 목적 세탁을 저절로 배웁니다.
"""

from __future__ import annotations
import math
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import elang as E

NAMES = ["철수", "영희", "민수", "지민", "서연", "하준", "도윤", "수아", "예준", "지우", "유나", "태오", "현우", "소윤"]
NAMES_OOD = ["마루", "누리", "다온", "라온", "새봄"]
ITEMS = ["사과", "귤", "사탕", "구슬", "연필", "스티커", "딱지", "동전", "쿠키", "젤리", "공책", "풍선"]
ITEMS_OOD = ["조약돌", "마나포션", "도토리", "솔방울"]

DISTRACTORS = [
    "{N}{은} {#d1}학년이다.",
    "오늘은 {#d1}월 {#d2}일이다.",
    "{N}{은} 파란 모자를 쓰고 있다.",
    "날씨가 아주 맑았다.",
    "{N}의 강아지는 {#d1}살이다.",
    "{N}{은} 버스를 {#d1}번 갈아탔다.",
    "창밖에서 새가 울고 있었다.",
]


def has_batchim(word: str) -> bool:
    ch = word[-1]
    if "가" <= ch <= "힣":
        return (ord(ch) - 0xAC00) % 28 != 0
    return ch in "013678"  # 숫자로 끝나는 경우 (영, 일, 삼, 육, 칠, 팔)


def josa(word: str, pair: str) -> str:
    a, b = {"은": ("은", "는"), "이": ("이", "가"), "을": ("을", "를"), "과": ("과", "와"), "으로": ("으로", "로")}[pair]
    return word + (a if has_batchim(word) else b)


@dataclass
class Fact:
    text: str
    terms: List[List[str]] = field(default_factory=list)  # 전위 토큰열들 (변수 기호, #숫자)


@dataclass
class Story:
    facts: List[Fact]
    question: str
    read: str
    var_types: Dict[str, str]
    values: Dict[str, float]  # 숫자 자리 값 + 변수 정답
    shuffle_facts: bool = False
    level: int = 1
    gen: str = ""


@dataclass
class Sample:
    text: str
    words: List[str]
    nums: List[float]
    tokens: List[str]
    truth: Dict[str, float]
    read_var: str
    level: int
    gen: str
    ood: bool


def fmt_num(v: float, rng: random.Random) -> str:
    if abs(v - round(v)) < 1e-9:
        iv = int(round(v))
        return f"{iv:,}" if iv >= 1000 and rng.random() < 0.6 else str(iv)
    return f"{v:.1f}".rstrip("0").rstrip(".")


# ======================================================================== 생성기들
def g_lin1(rng, names, items, ood) -> Story:
    x = rng.randint(2, 40)
    a = rng.randint(2, 9)
    b = rng.randint(1, 30)
    variants = [
        ("어떤 수에 {#a}{을} 곱하고 {#b}{을} 더하면 {#c}{이} 된다.", ["SQ", "SUB", "ADD", "MUL", "X", "#a", "#b", "#c"], a * x + b),
        ("어떤 수의 {#a}배보다 {#b}만큼 큰 수는 {#c}이다.", ["SQ", "SUB", "ADD", "MUL", "X", "#a", "#b", "#c"], a * x + b),
        ("어떤 수에 {#a}{을} 곱한 뒤 {#b}{을} 빼면 {#c}{이} 된다.", ["SQ", "SUB", "SUB", "MUL", "X", "#a", "#b", "#c"], a * x - b),
    ]
    held = [("어떤 수를 {#a}번 더하고 거기에 {#b}{을} 보탰더니 {#c}{이} 나왔다.", ["SQ", "SUB", "ADD", "MUL", "X", "#a", "#b", "#c"], a * x + b)]
    text, term, c = rng.choice(held if ood else variants)
    if c <= 0:
        c = a * x + b
        text, term = variants[0][0], variants[0][1]
    q = rng.choice(["[[X:어떤 수는]] 얼마인가?", "[[X:어떤 수를]] 구하시오.", "그렇다면 [[X:어떤 수는]] 무엇일까?"])
    return Story([Fact(text, [term])], q, "X", {"X": "INT"}, {"#a": a, "#b": b, "#c": c, "X": x}, level=1, gen="lin1")


def g_price(rng, names, items, ood) -> Story:
    it, nm = rng.choice(items), rng.choice(names)
    p = rng.choice([100, 150, 200, 250, 300, 400, 500, 700, 800, 1200])
    n = rng.randint(2, 12)
    texts = [f"{it} {{#n}}개를 사는 데 {{#t}}원이 들었다.", f"{josa(nm, '은')} {it} {{#n}}개를 {{#t}}원에 샀다."]
    held = [f"{it} {{#n}}개 값으로 {josa(nm, '은')} 모두 {{#t}}원을 냈다."]
    q = rng.choice([f"[[X:{it} 한 개의 가격은]] 얼마인가?", f"[[X:{it} 하나의 값은]] 몇 원인가?"])
    return Story([Fact(rng.choice(held if ood else texts), [["SQ", "SUB", "MUL", "X", "#n", "#t"]])], q, "X",
                 {"X": "INT"}, {"#n": n, "#t": n * p, "X": p}, level=1, gen="price")


def g_sumdiff(rng, names, items, ood) -> Story:
    A, B = rng.sample(names, 2)
    it = rng.choice(items)
    b = rng.randint(3, 40)
    d = rng.randint(1, 20)
    a = b + d
    f1 = rng.choice([
        f"[[A:{josa(A, '은')}]] [[B:{B}보다]] {it}{'을' if has_batchim(it) else '를'} {{#d}}개 더 많이 가지고 있다.",
        f"[[A:{A}의]] {it}{'은' if has_batchim(it) else '는'} [[B:{B}의]] 것보다 {{#d}}개 많다.",
    ] if not ood else [f"[[A:{josa(A, '은')}]] [[B:{B}에 비해]] {it} {{#d}}개를 더 갖고 있다."])
    f2 = rng.choice(["두 사람이 가진 " + josa(it, "은") + " 모두 {#s}개이다.", "둘의 " + josa(it, "을") + " 합치면 {#s}개가 된다."])
    read = rng.choice(["A", "B"])
    who = A if read == "A" else B
    q = f"[[{read}:{who}가 가진 {it}]]{'은' if has_batchim(it) else '는'} 몇 개인가?" if not has_batchim(who) else \
        f"[[{read}:{who}이 가진 {it}]]{'은' if has_batchim(it) else '는'} 몇 개인가?"
    facts = [Fact(f1, [["SQ", "SUB", "SUB", "A", "B", "#d"]]), Fact(f2, [["SQ", "SUB", "ADD", "A", "B", "#s"]])]
    return Story(facts, q, read, {"A": "INT", "B": "INT"}, {"#d": d, "#s": a + b, "A": a, "B": b},
                 shuffle_facts=True, level=2, gen="sumdiff")


def g_mult(rng, names, items, ood) -> Story:
    A, B = rng.sample(names, 2)
    it = rng.choice(items)
    b = rng.randint(2, 25)
    k = rng.randint(2, 6)
    f1 = (f"[[A:{A}의]] {josa(it, '은')} [[B:{B}의]] {{#k}}배이다." if not ood
          else f"[[A:{josa(A, '은')}]] [[B:{B}보다]] {it}{'을' if has_batchim(it) else '를'} {{#k}}배 많이 모았다.")
    f2 = rng.choice(["두 사람의 " + josa(it, "을") + " 모두 합하면 {#s}개이다.", "둘이 가진 " + josa(it, "은") + " 합쳐서 {#s}개다."])
    read = rng.choice(["A", "B"])
    who = A if read == "A" else B
    q = f"[[{read}:{who}의 {it}]]{'은' if has_batchim(it) else '는'} 몇 개인가?"
    facts = [Fact(f1, [["SQ", "SUB", "A", "MUL", "#k", "B"]]), Fact(f2, [["SQ", "SUB", "ADD", "A", "B", "#s"]])]
    return Story(facts, q, read, {"A": "INT", "B": "INT"}, {"#k": k, "#s": k * b + b, "A": k * b, "B": b},
                 shuffle_facts=False, level=2, gen="mult")


def g_combo(rng, names, items, ood) -> Story:
    X, Y = rng.sample(items, 2)
    px, py = rng.choice([100, 200, 300, 400, 500, 600]), rng.choice([150, 250, 350, 450, 700])
    a, b, c, d = rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 5)
    while a * d == b * c:
        d = rng.randint(1, 5)
    f1 = f"[[X:{X}]] {{#a}}개와 [[Y:{Y}]] {{#b}}개의 값은 {{#t1}}원이다."
    f2 = f"{X} {{#c}}개와 {Y} {{#d}}개를 사면 {{#t2}}원이다." if not ood else f"{X} {{#c}}개에 {Y} {{#d}}개를 더해 사니 {{#t2}}원을 냈다."
    read = rng.choice(["X", "Y"])
    q = f"{(X if read == 'X' else Y)} 한 개의 값은 얼마인가?"
    facts = [Fact(f1, [["SQ", "SUB", "ADD", "MUL", "#a", "X", "MUL", "#b", "Y", "#t1"]]),
             Fact(f2, [["SQ", "SUB", "ADD", "MUL", "#c", "X", "MUL", "#d", "Y", "#t2"]])]
    return Story(facts, q, read, {"X": "INT", "Y": "INT"},
                 {"#a": a, "#b": b, "#c": c, "#d": d, "#t1": a * px + b * py, "#t2": c * px + d * py, "X": px, "Y": py},
                 shuffle_facts=False, level=2, gen="combo")


def g_age(rng, names, items, ood) -> Story:
    A, B = rng.sample(names, 2)
    b, n, k = rng.randint(3, 15), rng.randint(2, 10), rng.randint(2, 3)
    a = k * (b + n) - n
    d = a - b
    f1 = f"[[A:{josa(A, '은')}]] [[B:{B}보다]] {{#d}}살 많다."
    f2 = f"{{#n}}년 뒤에는 {A}의 나이가 {B}의 나이의 {{#k}}배가 된다." if not ood else f"지금부터 {{#n}}년이 흐르면 {A}의 나이는 {B} 나이의 {{#k}}배가 된다."
    read = rng.choice(["A", "B"])
    q = f"지금 {(A if read == 'A' else B)}의 나이는 몇 살인가?"
    facts = [Fact(f1, [["SQ", "SUB", "SUB", "A", "B", "#d"]]),
             Fact(f2, [["SQ", "SUB", "ADD", "A", "#n", "MUL", "#k", "ADD", "B", "#n"]])]
    return Story(facts, q, read, {"A": "INT", "B": "INT"}, {"#d": d, "#n": n, "#k": k, "A": a, "B": b},
                 level=2, gen="age")


def g_speed(rng, names, items, ood) -> Story:
    nm = rng.choice(names)
    v, t = rng.choice([4, 5, 6, 12, 15, 40, 60, 80]), rng.randint(2, 6)
    mode = rng.choice(["v", "t", "d"])
    if mode == "v":
        f = f"{josa(nm, '은')} {{#d}}km를 {{#t}}시간 동안 일정한 빠르기로 이동했다." if not ood else f"{josa(nm, '은')} {{#t}}시간에 걸쳐 {{#d}}km를 같은 속도로 갔다."
        return Story([Fact(f, [["SQ", "SUB", "MUL", "X", "#t", "#d"]])], "[[X:이동한 빠르기는]] 시속 몇 km인가?", "X",
                     {"X": "REAL"}, {"#d": v * t, "#t": t, "X": v}, level=1, gen="speed")
    if mode == "t":
        f = f"{josa(nm, '은')} 시속 {{#v}}km로 {{#d}}km를 이동했다."
        return Story([Fact(f, [["SQ", "SUB", "MUL", "#v", "X", "#d"]])], "[[X:걸린 시간은]] 몇 시간인가?", "X",
                     {"X": "REAL"}, {"#v": v, "#d": v * t, "X": t}, level=1, gen="speed")
    f = f"{josa(nm, '은')} 시속 {{#v}}km로 {{#t}}시간 동안 쉬지 않고 이동했다."
    return Story([Fact(f, [["SQ", "SUB", "X", "MUL", "#v", "#t"]])], "[[X:이동한 거리는]] 몇 km인가?", "X",
                 {"X": "REAL"}, {"#v": v, "#t": t, "X": v * t}, level=1, gen="speed")


def g_height(rng, names, items, ood) -> Story:
    v = rng.choice([5, 8, 10, 12, 14, 15, 20, 24, 30])
    g = 9.8
    h = v * v / (2 * g)
    f = "공을 초속 {#v}m로 똑바로 위로 던졌다." if not ood else "돌멩이를 초속 {#v}m의 속력으로 하늘 위로 곧게 쏘아 올렸다."
    f2 = "중력가속도는 {#g}m/s²이다."
    return Story([Fact(f), Fact(f2, [["SQ", "SUB", "MUL", "MUL", "C2", "#g", "X", "SQ", "#v"]])],
                 "[[X:공이 올라가는 가장 높은 높이는]] 몇 m인가?" if not ood else "[[X:올라가는 최고 높이는]] 몇 m인가?",
                 "X", {"X": "REAL"}, {"#v": v, "#g": g, "X": h}, level=3, gen="height")


def g_ohm(rng, names, items, ood) -> Story:
    r, i = rng.choice([2, 4, 5, 10, 20, 50]), rng.choice([0.5, 1, 2, 3, 4])
    mode = rng.choice(["V", "I"])
    if mode == "V":
        f = "저항이 {#r}옴인 전선에 {#i}A의 전류가 흐른다."
        return Story([Fact(f, [["SQ", "SUB", "X", "MUL", "#i", "#r"]])], "[[X:전선에 걸린 전압은]] 몇 V인가?", "X",
                     {"X": "REAL"}, {"#r": r, "#i": i, "X": i * r}, level=1, gen="ohm")
    f = "저항이 {#r}옴인 전구에 {#v}V의 전압을 걸었다." if not ood else "{#v}V 전지에 {#r}옴짜리 저항을 연결했다."
    return Story([Fact(f, [["SQ", "SUB", "MUL", "X", "#r", "#v"]])], "[[X:흐르는 전류는]] 몇 A인가?", "X",
                 {"X": "REAL"}, {"#r": r, "#v": i * r, "X": i}, level=1, gen="ohm")


def g_salt(rng, names, items, ood) -> Story:
    s = rng.choice([10, 20, 25, 30, 40, 50, 60])
    w = rng.choice([150, 180, 200, 240, 300, 400, 450])
    f = "소금 {#s}g을 물 {#w}g에 녹여 소금물을 만들었다." if not ood else "물 {#w}g에 소금 {#s}g을 넣고 잘 저었다."
    return Story([Fact(f, [["SQ", "SUB", "MUL", "X", "ADD", "#s", "#w", "MUL", "C100", "#s"]])],
                 "[[X:이 소금물의 농도는]] 몇 %인가?", "X", {"X": "REAL"},
                 {"#s": s, "#w": w, "X": 100.0 * s / (s + w)}, level=2, gen="salt")


def g_boat(rng, names, items, ood) -> Story:
    k = rng.randint(3, 9)
    n = rng.randint(k + 1, k * 9)
    if n % k == 0:
        n += 1
    x = math.ceil(n / k)
    f1 = "한 번에 {#k}명씩 탈 수 있는 배가 있다." if not ood else "이 엘리베이터에는 한 번에 {#k}명까지만 탈 수 있다."
    f2 = "{#n}명이 모두 강을 건너려고 한다." if not ood else "{#n}명이 모두 위층으로 올라가야 한다."
    q = "[[X:배는 최소]] 몇 번 움직여야 하는가?" if not ood else "[[X:엘리베이터는 적어도]] 몇 번 올라가야 하는가?"
    terms = [["RSQ", "SUB", "#n", "MUL", "#k", "X"],
             ["RSQ", "SUB", "MUL", "#k", "SUB", "X", "C1", "SUB", "#n", "C1"]]
    return Story([Fact(f1), Fact(f2, terms)], q, "X", {"X": "INT"}, {"#k": k, "#n": n, "X": x}, level=3, gen="boat")


def g_chain(rng, names, items, ood) -> Story:
    A, B, C = rng.sample(names, 3)
    it = rng.choice(items)
    c, d, k = rng.randint(2, 15), rng.randint(1, 10), rng.randint(2, 4)
    b = c + d
    a = k * b
    f1 = f"[[A:{A}의]] {josa(it, '은')} [[B:{B}의]] {{#k}}배이다."
    f2 = f"{josa(B, '은')} [[C:{C}보다]] {it} {{#d}}개를 더 가지고 있다."
    f3 = f"{josa(C, '은')} {josa(it, '을')} {{#c}}개 가지고 있다."
    read = rng.choice(["A", "B"])
    q = f"{(A if read == 'A' else B)}의 {josa(it, '은')} 몇 개인가?"
    facts = [Fact(f1, [["SQ", "SUB", "A", "MUL", "#k", "B"]]), Fact(f2, [["SQ", "SUB", "SUB", "B", "C", "#d"]]),
             Fact(f3, [["SQ", "SUB", "C", "#c"]])]
    if ood:
        facts = facts[::-1]
    return Story(facts, q, read, {"A": "INT", "B": "INT", "C": "INT"},
                 {"#k": k, "#d": d, "#c": c, "A": a, "B": b, "C": c}, level=4, gen="chain")


def g_rect(rng, names, items, ood) -> Story:
    h, d = rng.randint(2, 20), rng.randint(1, 10)
    w = h + d
    f = "직사각형의 [[W:가로는]] [[H:세로보다]] {#d}cm 길고, 둘레는 {#p}cm이다." if not ood else \
        "어떤 직사각형의 둘레가 {#p}cm인데, [[W:가로가]] [[H:세로보다]] {#d}cm 더 길다."
    read = rng.choice(["W", "H"])
    q = "가로의 길이는 몇 cm인가?" if read == "W" else "세로의 길이는 몇 cm인가?"
    return Story([Fact(f, [["SQ", "SUB", "SUB", "W", "H", "#d"], ["SQ", "SUB", "MUL", "C2", "ADD", "W", "H", "#p"]])],
                 q, read, {"W": "INT", "H": "INT"}, {"#d": d, "#p": 2 * (w + h), "W": w, "H": h}, level=2, gen="rect")


GENERATORS: List[Callable] = [g_lin1, g_price, g_sumdiff, g_mult, g_combo, g_age, g_speed, g_height,
                              g_ohm, g_salt, g_boat, g_chain, g_rect]


# ======================================================================== 조립
_PH = re.compile(r"\{#(\w+)\}")
_JOSA_AFTER_PH = re.compile(r"\{(은|이|을|과)\}")
_LBL = re.compile(r"\[\[(\w+):(.+?)\]\]")


_JOSA = {"은": ("은", "는"), "이": ("이", "가"), "을": ("을", "를"), "과": ("과", "와")}


def _fill(text: str, values: Dict[str, float], rng: random.Random, order: List[str]) -> str:
    def rep(m):
        key = "#" + m.group(1)
        order.append(key)
        return fmt_num(values[key], rng)
    out = _PH.sub(rep, text)
    # {은}/{이}/{을}/{과} 자리: 바로 앞 글자의 받침을 보고 결정 (숫자, 이름 모두)
    while True:
        m = _JOSA_AFTER_PH.search(out)
        if not m:
            return out
        prev = out[: m.start()].replace("]]", "")
        j = _JOSA[m.group(1)][0 if prev and has_batchim(prev) else 1]
        out = out[: m.start()] + j + out[m.end():]


def assemble(story: Story, rng: random.Random, names: List[str], ood: bool, n_distract: int) -> Optional[Sample]:
    facts = list(story.facts)
    if story.shuffle_facts:
        rng.shuffle(facts)
    values = dict(story.values)
    # 방해 문장 끼워 넣기
    slots: List[Tuple[str, List[List[str]]]] = [(f.text, f.terms) for f in facts]
    for j in range(n_distract):
        t = rng.choice(DISTRACTORS).replace("{N}", rng.choice(names))
        for key in ("d1", "d2"):
            values[f"#{key}_{j}"] = rng.randint(1, 12)
            t = t.replace("{#" + key + "}", "{#" + key + "_" + str(j) + "}")
        slots.insert(rng.randint(0, len(slots)), (t, []))

    sentences, term_list, order = [], [], []
    for text, terms in slots:
        sentences.append(_fill(text, values, rng, order))
        term_list.extend(terms)
    sentences.append(_fill(story.question, values, rng, order))
    raw = " ".join(sentences)

    # 꼬리표 위치 찾기: [[X:구절]] 을 지우면서 단어 인덱스를 기록.
    # 꼬리표 바로 뒤에 공백 없이 붙은 조사("]]은")는 꼬리표 마지막 단어에 합칩니다.
    occurrences: Dict[str, List[List[int]]] = {}
    clean_words: List[str] = []
    pos = 0
    glue = False
    for m in list(_LBL.finditer(raw)) + [None]:
        seg = raw[pos: m.start()] if m else raw[pos:]
        if glue and seg and not seg.startswith(" "):
            first, _, seg = seg.partition(" ")
            clean_words[-1] += first
        clean_words.extend(seg.split())
        if m is None:
            break
        span_words = m.group(2).split()
        start = len(clean_words)
        if raw[m.start() - 1: m.start()] not in ("", " "):
            return None  # 꼬리표가 앞 단어에 붙어 있으면 단어 경계가 어긋남
        clean_words.extend(span_words)
        occurrences.setdefault(m.group(1), []).append(list(range(start, start + len(span_words))))
        pos = m.end()
        glue = True
    text = " ".join(clean_words)
    # 이름표: 답 변수는 질문 문장 속 구절(마지막 등장), 나머지는 처음 등장한 구절
    labels = {v: (occ[-1] if v == story.read else occ[0]) for v, occ in occurrences.items()}
    words = E.split_words(text)
    if len(words) > E.MAX_WORDS:
        return None
    for var in story.var_types:
        if var not in labels:
            # 꼬리표가 템플릿에 없으면 질문 문장 첫 단어를 이름표로 사용
            return None

    nums, _ = E.extract_numbers(words)
    expect = [values[k] for k in order]
    if len(nums) != len(expect) or any(abs(a - b) > 1e-9 for a, b in zip(nums, expect)):
        return None
    num_index = {}
    for idx, key in enumerate(order):
        num_index.setdefault(key, idx)

    # 변수 순서 = 원문에 처음 등장하는 순서 (정준형). 질문 문장을 끝까지 안 읽어도 이름을 정할 수 있게.
    var_order = sorted(story.var_types, key=lambda v: occurrences[v][0][0])
    vmap = {v: f"v{i}" for i, v in enumerate(var_order)}
    toks = ["BOS"]
    for v in var_order:
        toks += ["VAR", vmap[v], story.var_types[v]]
    for term in term_list:
        toks.append("TERM")
        for t in term:
            if t.startswith("#"):
                toks.append(f"N{num_index[t]}")
            elif t in vmap:
                toks.append(vmap[t])
            else:
                toks.append(t)
    toks += ["READ", vmap[story.read]]
    for v in var_order:
        toks += ["LBL", vmap[v]] + [f"W{i}" for i in labels[v][: E.MAX_LABEL_WORDS]]
    toks.append("EOS")

    truth = {vmap[v]: float(story.values[v]) for v in story.var_types}
    return Sample(text=text, words=words, nums=nums, tokens=toks, truth=truth, read_var=vmap[story.read],
                  level=story.level, gen=story.gen, ood=ood)


def make_sample(rng: random.Random, ood: bool = False, gen: Optional[Callable] = None) -> Sample:
    names = NAMES_OOD if ood else NAMES
    items = ITEMS_OOD if ood else ITEMS
    while True:
        g = gen or rng.choice(GENERATORS)
        story = g(rng, names, items, ood)
        n_distract = rng.choice([0, 0, 1, 1, 2])
        s = assemble(story, rng, names, ood, n_distract)
        if s is None:
            continue
        chk = E.checkpoints(s.tokens, s.nums, len(s.words), s.truth, s.read_var, s.words)
        if all(chk.values()):
            return s


if __name__ == "__main__":
    rng = random.Random(0)
    fails = 0
    for g in GENERATORS:
        s = make_sample(rng, gen=g)
        print(f"\n=== {g.__name__} (L{s.level}) ===")
        print(s.text)
        print(E.render_program(E.parse(s.tokens), s.nums, s.words))
        print("정답:", {k: round(v, 3) for k, v in s.truth.items()}, "READ", s.read_var)
    print("\n=== OOD 예시 ===")
    for _ in range(3):
        s = make_sample(rng, ood=True)
        print(s.gen, "|", s.text)
