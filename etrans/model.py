"""
E-CLIP 번역기 본체.

👂 귀     : 사전학습 다국어 문장 인코더 (multilingual-e5-small, 얼려서 사용)
🔗 귀 보정 : 귀 출력 + 단어 번호 + 숫자 슬롯 번호를 섞는 작은 트랜스포머 (학습)
👁️ 수식 눈 : E-Lang 수식 인코더 (처음부터 학습, 트리 깊이 임베딩)
💍 결혼식장 : 문장 ↔ 수식 CLIP 대조 학습 (한 끗 차이 가짜 신랑 포함)
✍️ 손     : E-Lang 토큰을 쓰는 디코더 (처음부터 학습, 문법 강제 디코딩)

숫자와 꼬리표는 원문 위치 포인터(N_k, W_k)로만 쓰므로, 귀 출력에 "이 조각은 몇 번째 단어/숫자"
정보를 더해 주면 손이 그 자리를 보고 해당 포인터 토큰을 고르는 법을 배웁니다.
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import elang as E

EAR_NAME = "intfloat/multilingual-e5-small"
EAR_DIM = 384
MAX_SUBWORDS = 160
MAX_TGT = 96


class Ear:
    """얼린 귀. 학습 중에는 결과를 미리 계산해 캐시합니다."""

    def __init__(self, cache_dir: str = "/home/user/models", device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(EAR_NAME, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(EAR_NAME, cache_dir=cache_dir).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, word_lists: List[List[str]]) -> List[Dict[str, torch.Tensor]]:
        enc = self.tok(word_lists, is_split_into_words=True, truncation=True, max_length=MAX_SUBWORDS,
                       padding=True, return_tensors="pt").to(self.device)
        h = self.model(**enc).last_hidden_state.float().cpu()
        out = []
        for b, words in enumerate(word_lists):
            wids = enc.word_ids(b)
            _, slot_of_word = E.extract_numbers(words)
            L = int(enc["attention_mask"][b].sum())
            word_id = torch.tensor([(w if w is not None and w < E.MAX_WORDS else -1) for w in wids[:L]])
            num_slot = torch.tensor([(slot_of_word[w] if w is not None and w < len(slot_of_word) else -1) for w in wids[:L]])
            out.append({"h": h[b, :L].half(), "word_id": word_id, "num_slot": num_slot})
        return out


def formula_depths(tokens: Sequence[str]) -> List[int]:
    """전위 수식 토큰마다 트리 깊이를 계산 (수식 눈의 트리 위치 정보)."""
    depths, stack = [], []  # stack: 열려 있는 조상마다 남은 자식 칸 수
    for t in tokens:
        if t == "TERM":
            stack = []
            depths.append(0)
        elif t in E.ARITY or t in E.VARS or t in E.NUMS or t in E.CONSTS:
            depths.append(min(len(stack) + 1, 31))
            if stack:
                stack[-1] -= 1
                while stack and stack[-1] == 0:
                    stack.pop()
            if E.ARITY.get(t, 0):
                stack.append(E.ARITY[t])
        else:
            depths.append(0)
    return depths


def formula_part(tokens: Sequence[str]) -> List[str]:
    """CLIP 수식 눈이 보는 부분 = 추론기가 받는 순수 수식 (장부 포인터 제거)."""
    return E.formula_only(tokens)


class ECLIP(nn.Module):
    def __init__(self, d: int = 256, n_heads: int = 4, n_dec: int = 4, n_mem: int = 2, n_eye: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        V = len(E.VOCAB)
        self.d = d
        # 귀 보정
        self.ear_proj = nn.Linear(EAR_DIM, d)
        self.word_emb = nn.Embedding(E.MAX_WORDS + 1, d)
        self.num_emb = nn.Embedding(E.MAX_NUMS + 1, d)
        layer = lambda: nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.mem_enc = nn.TransformerEncoder(layer(), n_mem)
        # 손
        self.tok_emb = nn.Embedding(V, d)
        self.pos_emb = nn.Embedding(MAX_TGT + 8, d)
        dec_layer = nn.TransformerDecoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec)
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, V)
        # 내용을 품은 포인터: W_k / N_k 토큰에 해당 단어·숫자의 귀 벡터를 더하고, 고를 때도 내용으로 점수를 매김
        self.ptr_in = nn.Linear(d, d)
        self.ptr_q = nn.Linear(d, d)
        self.ptr_k = nn.Linear(d, d)
        self.W0, self.N0 = E.TOK2ID["W0"], E.TOK2ID["N0"]
        # 변수 포인터: 답 구절 내용 vs 각 변수 이름표 내용을 직접 비교해 READ 변수 선택에 더함
        self.vp_a = nn.Linear(d, d)
        self.vp_v = nn.Linear(d, d)
        self.V0 = E.TOK2ID["v0"]
        self.VAR_ID, self.TERM_ID, self.READ_ID = E.TOK2ID["VAR"], E.TOK2ID["TERM"], E.TOK2ID["READ"]
        # 수식 눈 + 결혼식장
        self.eye_tok = nn.Embedding(V, d)
        self.eye_depth = nn.Embedding(32, d)
        self.eye_pos = nn.Embedding(MAX_TGT + 8, d)
        self.eye = nn.TransformerEncoder(layer(), n_eye)
        self.text_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.formula_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    # ------------------------------------------------------------ 귀
    def memory(self, h, word_id, num_slot, mem_mask):
        x = self.ear_proj(h.float()) + self.word_emb(word_id + 1) + self.num_emb(num_slot + 1)
        return self.mem_enc(x, src_key_padding_mask=~mem_mask)

    # ------------------------------------------------------------ 포인터 표
    @staticmethod
    def _slot_mean(mem, slot, n_slots):
        """조각(subword)들의 귀 벡터를 단어/숫자 슬롯별로 평균 -> [B, n_slots, d]"""
        B, L, d = mem.shape
        valid = (slot >= 0) & (slot < n_slots)
        idx = slot.clamp(min=0, max=n_slots - 1)
        out = torch.zeros(B, n_slots, d, dtype=mem.dtype)
        cnt = torch.zeros(B, n_slots, 1, dtype=mem.dtype)
        w = valid.unsqueeze(-1).to(mem.dtype)
        out.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, d), mem * w)
        cnt.scatter_add_(1, idx.unsqueeze(-1), w)
        return out / cnt.clamp(min=1.0)

    def pointer_tables(self, mem, word_id, num_slot):
        return self._slot_mean(mem, word_id, E.MAX_WORDS), self._slot_mean(mem, num_slot, E.MAX_NUMS)

    def var_pointer_bonus(self, tgt_in, word_vecs):
        """READ 뒤 위치마다 [B, T, MAX_VARS] 점수. 답 구절(지금까지 짚은 W)과 변수 이름표(W)의 내용 유사도."""
        K, V = E.MAX_WORDS, E.MAX_VARS
        is_w = (tgt_in >= self.W0) & (tgt_in < self.W0 + K)
        w_oh = F.one_hot((tgt_in - self.W0).clamp(0, K - 1), K).float() * is_w.unsqueeze(-1)
        after_read = torch.cumsum((tgt_in == self.READ_ID).int(), 1) > 0
        before_term = torch.cumsum((tgt_in == self.TERM_ID).int(), 1) == 0
        var_idx = torch.cumsum((tgt_in == self.VAR_ID).int(), 1) - 1
        in_label = (before_term & (var_idx >= 0)).unsqueeze(-1).float()
        var_oh = F.one_hot(var_idx.clamp(0, V - 1), V).float() * in_label
        M = torch.einsum("btv,btk->bvk", var_oh, w_oh)  # 이름표는 READ 전에 다 쓰이므로 전체 열에서 모아도 인과성 유지
        A = torch.cumsum(w_oh * after_read.unsqueeze(-1).float(), 1)
        var_vec = (M @ word_vecs) / M.sum(-1, keepdim=True).clamp(min=1)
        ans_vec = (A @ word_vecs) / A.sum(-1, keepdim=True).clamp(min=1)
        score = torch.einsum("btd,bvd->btv", self.vp_a(ans_vec), self.vp_v(var_vec)) / math.sqrt(word_vecs.size(-1))
        active = (after_read & (A.sum(-1) > 0)).unsqueeze(-1).float()
        return score * active

    # ------------------------------------------------------------ 손
    def decode_logits(self, tgt_in, mem, mem_mask, word_vecs, num_vecs):
        T = tgt_in.size(1)
        pos = torch.arange(T, device=tgt_in.device)
        y = self.tok_emb(tgt_in) + self.pos_emb(pos)[None]
        # 입력 쪽: 포인터 토큰에 가리키는 내용 더하기
        is_w = (tgt_in >= self.W0) & (tgt_in < self.W0 + E.MAX_WORDS)
        is_n = (tgt_in >= self.N0) & (tgt_in < self.N0 + E.MAX_NUMS)
        wi = (tgt_in - self.W0).clamp(0, E.MAX_WORDS - 1)
        ni = (tgt_in - self.N0).clamp(0, E.MAX_NUMS - 1)
        w_content = torch.gather(word_vecs, 1, wi.unsqueeze(-1).expand(-1, -1, word_vecs.size(-1)))
        n_content = torch.gather(num_vecs, 1, ni.unsqueeze(-1).expand(-1, -1, num_vecs.size(-1)))
        y = y + self.ptr_in(w_content * is_w.unsqueeze(-1) + n_content * is_n.unsqueeze(-1))
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), 1)
        y = self.decoder(y, mem, tgt_mask=causal, tgt_key_padding_mask=(tgt_in == E.PAD_ID),
                         memory_key_padding_mask=~mem_mask)
        h = self.out_norm(y)
        logits = self.out(h)
        # 출력 쪽: 포인터 토큰 점수에 "지금 원하는 내용 · 그 자리 내용" 유사도를 더함
        q = self.ptr_q(h) / math.sqrt(h.size(-1))
        ptr_w = torch.einsum("btd,bkd->btk", q, self.ptr_k(word_vecs))
        ptr_n = torch.einsum("btd,bkd->btk", q, self.ptr_k(num_vecs))
        bonus = torch.zeros_like(logits)
        bonus[..., self.W0: self.W0 + E.MAX_WORDS] = ptr_w
        bonus[..., self.N0: self.N0 + E.MAX_NUMS] = ptr_n
        bonus[..., self.V0: self.V0 + E.MAX_VARS] = self.var_pointer_bonus(tgt_in, word_vecs)
        return logits + bonus

    # ------------------------------------------------------------ 결혼식장
    def text_embed(self, mem, mem_mask):
        m = mem_mask.unsqueeze(-1).float()
        pooled = (mem * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.text_head(pooled), dim=-1)

    def formula_embed(self, f_ids, f_depth):
        T = f_ids.size(1)
        pos = torch.arange(T, device=f_ids.device)
        x = self.eye_tok(f_ids) + self.eye_depth(f_depth) + self.eye_pos(pos)[None]
        mask = f_ids != E.PAD_ID
        x = self.eye(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(self.formula_head(pooled), dim=-1)


# ==================================================================== 한 끗 차이 가짜 신랑
def mutate_formula(tokens: List[str], n_nums: int, rng: random.Random) -> Optional[List[str]]:
    """정답 수식과 거의 같지만 의미가 다른 수식을 만듭니다 (부호, 숫자 포인터, 변수, 항 개수)."""
    f = list(tokens)
    for _ in range(10):
        g = list(f)
        kind = rng.choice(["num", "op", "var", "drop"])
        if kind == "num":
            idx = [i for i, t in enumerate(g) if t in E.NUMS]
            if idx and n_nums > 1:
                i = rng.choice(idx)
                g[i] = rng.choice([n for n in E.NUMS[:n_nums] if n != g[i]])
        elif kind == "op":
            idx = [i for i, t in enumerate(g) if t in ("ADD", "SUB", "MUL")]
            if idx:
                i = rng.choice(idx)
                g[i] = rng.choice([o for o in ("ADD", "SUB", "MUL") if o != g[i]])
        elif kind == "var":
            nv = sum(1 for t in g if t == "VAR")
            idx = [i for i, t in enumerate(g) if t in E.VARS and i > 0 and g[i - 1] not in ("VAR", "READ")]
            if idx and nv > 1:
                i = rng.choice(idx)
                g[i] = rng.choice([v for v in E.VARS[:nv] if v != g[i]])
        else:
            starts = [i for i, t in enumerate(g) if t == "TERM"]
            if len(starts) > 1:
                s = rng.choice(starts)
                e = next((j for j in range(s + 1, len(g)) if g[j] in ("TERM", "READ")), len(g))
                g = g[:s] + g[e:]
        if g != f:
            return g
    return None
