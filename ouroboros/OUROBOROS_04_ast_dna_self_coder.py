"""
================================================================================
PROJECT OUROBOROS | MODULE 4: HUXLEY-GÖDEL PYTHON AST DNA SELF-CODER
File: OUROBOROS_04_ast_dna_self_coder.py
================================================================================
세계관 및 아키텍처 매핑 (Tearing Off the Wallpaper & Hacking Its Own DNA Code):
- 파이썬 내장 `ast` (Abstract Syntax Tree) 모듈을 사용하여, 뇌세포가 근육을 고치는 함수와
  다림질 물리 법칙 함수의 **실제 파이썬 소스코드 구문 트리(AST)**를 파싱하고,
  새로운 수식 줄(Statement)과 비선형 연산 노드를 삽입·변이·교배합니다.
- 승리한 코드는 디스크(`OUROBOROS_evolved_dna_kernel.py`)에 실제 파이썬 파일로 영구 기록됩니다!

[v2 업그레이드 ① 유전자 뷔페 무한 리필기]
- 고정된 8개 유전자 문장을 다 먹고 357노드에서 멈추던 문제를 해결합니다.
- GeneMutator(ast.NodeTransformer)가 기존 유전자를 점 돌연변이(상수 흔들기, tanh/sin/arctan 교체,
  +/- 뒤집기)시켜 '새 유전자'를 발명하고, 그 유전자를 쓴 후보가 챔피언이 되면 유전자 은행에 영구 등록합니다.
- 엘리트 보존: 부모도 같은 평가 가시 산에서 다시 채점받아, 자식이 못하면 부모 커널을 그대로 유지합니다.
- 적합도는 족보 OFF(allow_oracle_peek=False)로 채점합니다. 진짜 실력으로만 살아남기.
- 계보 기록: 챔피언이 바뀔 때마다 dna_lineage.jsonl 에 소스코드째 한 줄씩 남깁니다.
================================================================================
"""

from __future__ import annotations
import ast
import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np


PRIMORDIAL_DNA_SOURCE = '''import numpy as np

def evolved_plasticity_kernel(W_fast, k_vec, v_vec, spike_intensity):
    """[Gen 0 Primordial Amoeba DNA] Basic linear Hebbian delta muscle update."""
    wk = W_fast @ k_vec
    err = v_vec - wk
    gain = 0.16
    delta_fast = gain * np.outer(err, k_vec)
    W_next = 0.92 * W_fast + delta_fast
    return np.clip(W_next, -1.8, 1.8).astype(np.float32)

def evolved_ironing_kernel(best_mold, grad, brain_delta, stroke, n_presses):
    """[Gen 0 Primordial Amoeba DNA] Basic plastic-spoon linear steam ironing."""
    step_lr = 0.30 / (1.0 + 0.12 * stroke)
    proposal = best_mold - step_lr * grad + 0.12 * brain_delta
    return np.clip(proposal, -2.5, 2.5).astype(np.float32)
'''


PLASTICITY_AST_GENES = [
    "err = np.tanh(err + 0.55 * (err ** 3))",
    "W_next = W_next - 0.045 * float(np.mean(v_vec ** 2)) * W_fast",
    "W_next = W_next + 0.085 * np.tanh(W_fast @ W_fast.T) @ delta_fast",
    "W_next = W_next + min(0.25, float(spike_intensity) * 0.4) * delta_fast",
]

IRONING_AST_GENES = [
    "proposal = proposal - (0.08 / (1.0 + 0.05 * stroke)) * np.sin(grad * 2.5)",
    "proposal = proposal - min(0.32, 0.035 * n_presses) * grad",
    "proposal = proposal - 0.18 * np.tanh(grad + 0.6 * (grad ** 3)) + 0.15 * brain_delta",
    "proposal = 0.85 * proposal + 0.15 * np.tanh(best_mold - 0.55 * grad + 0.25 * brain_delta)",
]

# 변이로 바꿔 끼울 수 있는 '안전한' 비선형 함수들 (모두 유계 또는 완만함)
SWAPPABLE_NP_FUNCS = ["tanh", "sin", "arctan"]
MAX_GENES_PER_KERNEL = 8


class GeneMutator(ast.NodeTransformer):
    """유전자 한 줄(Assign 문)의 AST를 점 돌연변이시키는 외과 수술 도구."""

    def __init__(self, rng: np.random.Generator, rate: float = 0.45):
        self.rng = rng
        self.rate = rate
        self.n_mutations = 0

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, float) and self.rng.uniform() < self.rate:
            factor = float(np.exp(self.rng.normal(0.0, 0.3)))
            node.value = round(node.value * factor, 4)
            self.n_mutations += 1
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr in SWAPPABLE_NP_FUNCS
            and self.rng.uniform() < self.rate * 0.35
        ):
            others = [f for f in SWAPPABLE_NP_FUNCS if f != node.attr]
            node.attr = str(self.rng.choice(others))
            self.n_mutations += 1
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.Add, ast.Sub)) and self.rng.uniform() < self.rate * 0.15:
            node.op = ast.Sub() if isinstance(node.op, ast.Add) else ast.Add()
            self.n_mutations += 1
        return node


@dataclass
class DNACodeOrganism:
    dna_id: str
    generation: int
    parent_ids: List[str]
    source_code: str
    p_gene_src: List[str] = field(default_factory=list)
    i_gene_src: List[str] = field(default_factory=list)
    novel_genes: List[Tuple[str, str]] = field(default_factory=list)  # (kind, src) 이 후보가 발명한 새 유전자
    immediate_fitness: float = 0.0
    descendant_cmp_score: float = 0.0
    ast_node_count: int = 0


class HuxleyGodelASTSelfCoder:
    def __init__(
        self,
        output_dir: str = ".",
        seed: int = 2026,
        n_candidates: int = 5,
        max_archive: int = 256,
    ):
        self.output_dir = output_dir
        self.rng = np.random.default_rng(seed)
        self.n_candidates = n_candidates
        self.max_archive = max_archive
        self.clade_archive: List[DNACodeOrganism] = []
        self.champion: Optional[DNACodeOrganism] = None
        self.dna_file_path = os.path.join(output_dir, "OUROBOROS_evolved_dna_kernel.py")
        self.lineage_path = os.path.join(output_dir, "dna_lineage.jsonl")

        # 유전자 은행 (인스턴스마다 따로 자라남)
        self.plasticity_genes: List[str] = list(PLASTICITY_AST_GENES)
        self.ironing_genes: List[str] = list(IRONING_AST_GENES)
        self.gene_births: List[Dict[str, Any]] = []
        self.total_mutation_attempts = 0
        self.total_novel_genes_committed = 0

        self._bootstrap_primordial_dna()

    # ------------------------------------------------------------------ 체크포인트
    def __getstate__(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def relocate(self, output_dir: str) -> None:
        """체크포인트를 다른 폴더(예: 다음 Kaggle 세션)에서 이어받을 때 출력 경로를 옮깁니다."""
        self.output_dir = output_dir
        self.dna_file_path = os.path.join(output_dir, "OUROBOROS_evolved_dna_kernel.py")
        self.lineage_path = os.path.join(output_dir, "dna_lineage.jsonl")

    # ------------------------------------------------------------------ 탄생
    def _bootstrap_primordial_dna(self) -> None:
        tree = ast.parse(PRIMORDIAL_DNA_SOURCE)
        org0 = DNACodeOrganism(
            dna_id="dna_gen0_amoeba_seed",
            generation=0,
            parent_ids=[],
            source_code=ast.unparse(tree),
            immediate_fitness=0.1,
            descendant_cmp_score=0.1,
            ast_node_count=sum(1 for _ in ast.walk(tree)),
        )
        self.clade_archive.append(org0)
        self.champion = org0
        self.save_active_dna_to_disk(org0)

    @property
    def gene_bank_size(self) -> int:
        return len(self.plasticity_genes) + len(self.ironing_genes)

    # ------------------------------------------------------------------ AST 조립
    def build_ast_from_genes(
        self, p_genes: List[str], i_genes: List[str], gen_idx: int, dna_id: str
    ) -> Tuple[str, int]:
        tree = ast.parse(PRIMORDIAL_DNA_SOURCE)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "evolved_plasticity_kernel":
                genes, label = p_genes, "Self-Coded SRWM Muscle Plasticity Kernel"
            elif isinstance(node, ast.FunctionDef) and node.name == "evolved_ironing_kernel":
                genes, label = i_genes, "Self-Coded Plasma Steam-Iron Kernel"
            else:
                continue
            node.body[0] = ast.Expr(
                value=ast.Constant(value=f"[{dna_id} | Gen {gen_idx}] {label} ({len(genes)} genes)")
            )
            ret_stmt = node.body.pop()
            for src in genes:
                node.body.append(ast.parse(src).body[0])
            node.body.append(ret_stmt)

        ast.fix_missing_locations(tree)
        new_source = ast.unparse(tree)
        node_count = sum(1 for _ in ast.walk(tree))
        return new_source, node_count

    # ------------------------------------------------------------------ 점 돌연변이
    def mutate_gene(self, gene_src: str) -> Optional[str]:
        self.total_mutation_attempts += 1
        tree = ast.parse(gene_src)
        stmt = tree.body[0]
        if not isinstance(stmt, ast.Assign):
            return None
        target = stmt.targets[0]
        mutator = GeneMutator(self.rng)
        stmt.value = mutator.visit(stmt.value)
        if mutator.n_mutations == 0:
            # 아무것도 안 바뀌었으면 상수 하나는 무조건 흔든다
            consts = [n for n in ast.walk(stmt.value) if isinstance(n, ast.Constant) and isinstance(n.value, float)]
            if not consts:
                return None
            c = consts[int(self.rng.integers(0, len(consts)))]
            c.value = round(c.value * float(np.exp(self.rng.normal(0.0, 0.3))), 4)
        stmt.targets = [target]
        ast.fix_missing_locations(tree)
        new_src = ast.unparse(tree).strip()
        if new_src == gene_src or new_src in self.plasticity_genes or new_src in self.ironing_genes:
            return None
        return new_src

    # ------------------------------------------------------------------ 샌드박스
    def compile_and_sandbox_dna(
        self, source_code: str
    ) -> Tuple[bool, Optional[Callable], Optional[Callable], str]:
        try:
            code_obj = compile(source_code, filename="<self_coded_ast_dna>", mode="exec")
            sandbox_ns: Dict[str, Any] = {}
            exec(code_obj, sandbox_ns)

            p_fn = sandbox_ns["evolved_plasticity_kernel"]
            i_fn = sandbox_ns["evolved_ironing_kernel"]

            W_test = np.eye(8, dtype=np.float32) * 0.2
            k_test = np.ones(8, dtype=np.float32) / math.sqrt(8)
            v_test = np.zeros(8, dtype=np.float32)
            W_out = p_fn(W_test, k_test, v_test, 0.25)
            assert W_out.shape == (8, 8) and np.all(np.isfinite(W_out))

            m_test = np.zeros((4, 4), dtype=np.float32)
            g_test = np.ones((4, 4), dtype=np.float32) * 0.1
            d_test = np.zeros((4, 4), dtype=np.float32)
            m_out = i_fn(m_test, g_test, d_test, 1, 2)
            assert m_out.shape == (4, 4) and np.all(np.isfinite(m_out))

            # 무작위 큰 입력에서도 폭주하지 않는지 한 번 더 확인
            rs = np.random.default_rng(7)
            W_r = rs.normal(0, 1.0, (12, 12)).astype(np.float32)
            k_r = rs.normal(0, 1.0, 12).astype(np.float32)
            v_r = rs.normal(0, 1.0, 12).astype(np.float32)
            assert np.all(np.isfinite(p_fn(W_r, k_r, v_r, 3.0)))
            m_r = rs.normal(0, 1.5, (16, 8)).astype(np.float32)
            assert np.all(np.isfinite(i_fn(m_r, m_r * 0.3, m_r * 0.1, 17, 500)))

            return True, p_fn, i_fn, "SANDBOX_PASS"
        except Exception as e:
            return False, None, None, f"SANDBOX_ERROR: {e}"

    # ------------------------------------------------------------------ 진화 한 세대
    def _cap_genes(self, genes: List[str]) -> List[str]:
        genes = list(dict.fromkeys(genes))
        while len(genes) > MAX_GENES_PER_KERNEL:
            genes.pop(int(self.rng.integers(0, len(genes))))
        return genes

    def _propose_candidate(
        self, c_idx: int, top_parent: DNACodeOrganism, parents: List[DNACodeOrganism]
    ) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
        p_genes = list(top_parent.p_gene_src)
        i_genes = list(top_parent.i_gene_src)
        novel: List[Tuple[str, str]] = []

        if c_idx in (0, 1):
            # 은행에서 아직 안 쓴 유전자 하나씩 꺼내 붙이기 (원조 방식)
            avail_p = [g for g in self.plasticity_genes if g not in p_genes]
            avail_i = [g for g in self.ironing_genes if g not in i_genes]
            if avail_p and (self.rng.uniform() < 0.75 or not p_genes):
                p_genes.append(avail_p[int(self.rng.integers(0, len(avail_p)))])
            if avail_i and (self.rng.uniform() < 0.80 or not i_genes):
                i_genes.append(avail_i[int(self.rng.integers(0, len(avail_i)))])
            if not avail_p and not avail_i and (p_genes or i_genes):
                # 다 먹었으면 하나 뱉어보기 (가지치기 탐색)
                pool = p_genes if (p_genes and self.rng.uniform() < 0.5) or not i_genes else i_genes
                pool.pop(int(self.rng.integers(0, len(pool))))
        elif c_idx == 2 and len(parents) >= 2:
            # 교배: 2등 부모의 유전자와 합치기
            p_genes = p_genes + [g for g in parents[1].p_gene_src if g not in p_genes]
            i_genes = i_genes + [g for g in parents[1].i_gene_src if g not in i_genes]
        else:
            # 신규 유전자 발명: 기존 유전자를 점 돌연변이
            for kind, bank, active in (
                ("plasticity", self.plasticity_genes, p_genes),
                ("ironing", self.ironing_genes, i_genes),
            ):
                if self.rng.uniform() < 0.6 or kind == "ironing":
                    source_pool = active if (active and self.rng.uniform() < 0.7) else bank
                    src = source_pool[int(self.rng.integers(0, len(source_pool)))]
                    mutant = self.mutate_gene(src)
                    if mutant is None:
                        continue
                    novel.append((kind, mutant))
                    if src in active and self.rng.uniform() < 0.5:
                        active[active.index(src)] = mutant  # 대체 변이
                    else:
                        active.append(mutant)  # 중복 후 변이

        return self._cap_genes(p_genes), self._cap_genes(i_genes), novel

    def _score_on_spikes(self, brain: Any, solver: Any, eval_spikes: List[Any], gen: int,
                         p_fn: Callable, i_fn: Callable, n_nodes: int) -> Tuple[float, float, float]:
        brain.dna_plasticity_kernel = p_fn
        brain.dna_ironing_kernel = i_fn
        energies, strokes = [], []
        for sp in eval_spikes:
            res = solver.iron_spike_to_flat_floor(
                sp,
                max_ironing_strokes=10,
                n_superposition_irons=4,
                current_epoch=gen,
                install_press_on_success=False,
                allow_oracle_peek=False,  # 족보 없이 진짜 실력으로 채점
                allow_mitosis=False,      # 후보 채점 중에는 팔을 늘리지 않음
            )
            energies.append(res["final_spike_height"])
            strokes.append(res["strokes_used"])
        mean_e = float(np.mean(energies))
        mean_s = float(np.mean(strokes))
        # 원본 적합도에서 '유전자 개수 보너스'를 빼고 작은 간결성(parsimony) 페널티로 교체:
        # 유전자 은행이 무한히 자라므로 코드가 쓸데없이 비대해지는 것을 막습니다.
        fitness = (1.0 / (mean_e + 0.005)) - 0.15 * mean_s - 0.002 * n_nodes
        return fitness, mean_e, mean_s

    def evolve_and_hotswap_dna(
        self,
        brain: Any,
        solver: Any,
        eval_spikes: List[Any],
        next_gen: int,
    ) -> Dict[str, Any]:
        parents = sorted(
            self.clade_archive,
            key=lambda d: 0.6 * d.immediate_fitness + 0.4 * d.descendant_cmp_score,
            reverse=True,
        )
        top_parent = parents[0]

        candidates: List[DNACodeOrganism] = []
        for c_idx in range(self.n_candidates):
            p_genes, i_genes, novel = self._propose_candidate(c_idx, top_parent, parents)
            dna_id = f"dna_gen{next_gen}_v{c_idx}_{int(self.rng.integers(100, 999))}"
            src, n_nodes = self.build_ast_from_genes(p_genes, i_genes, next_gen, dna_id)
            candidates.append(
                DNACodeOrganism(
                    dna_id=dna_id,
                    generation=next_gen,
                    parent_ids=[top_parent.dna_id],
                    source_code=src,
                    p_gene_src=p_genes,
                    i_gene_src=i_genes,
                    novel_genes=novel,
                    ast_node_count=n_nodes,
                )
            )

        # 엘리트 보존: 현재 챔피언도 같은 가시 산으로 재채점
        incumbent = self.champion or top_parent
        ok, p_inc, i_inc, _ = self.compile_and_sandbox_dna(incumbent.source_code)
        best_org, best_fns, best_score = incumbent, None, -1e9
        incumbent_fitness = None
        if ok:
            incumbent_fitness, _, _ = self._score_on_spikes(
                brain, solver, eval_spikes, next_gen, p_inc, i_inc, incumbent.ast_node_count
            )
            best_score, best_fns = incumbent_fitness, (p_inc, i_inc)

        sandbox_failures = 0
        for cand in candidates:
            ok, p_fn, i_fn, _ = self.compile_and_sandbox_dna(cand.source_code)
            if not ok:
                sandbox_failures += 1
                continue
            fitness, _, _ = self._score_on_spikes(
                brain, solver, eval_spikes, next_gen, p_fn, i_fn, cand.ast_node_count
            )
            cand.immediate_fitness = fitness
            cand.descendant_cmp_score = fitness * (1.0 + 0.04 * len(cand.p_gene_src))
            self.clade_archive.append(cand)

            if fitness > best_score:
                best_score = fitness
                best_org = cand
                best_fns = (p_fn, i_fn)

        if candidates:
            child_best = max((c.immediate_fitness for c in candidates), default=0.0)
            top_parent.descendant_cmp_score = 0.5 * top_parent.descendant_cmp_score + 0.5 * child_best

        champion_changed = best_org is not incumbent
        committed_genes: List[str] = []
        if champion_changed:
            for kind, src in best_org.novel_genes:
                bank = self.plasticity_genes if kind == "plasticity" else self.ironing_genes
                if src in (best_org.p_gene_src + best_org.i_gene_src) and src not in bank:
                    bank.append(src)
                    self.total_novel_genes_committed += 1
                    committed_genes.append(src)
                    self.gene_births.append({"kind": kind, "source": src, "born_gen": next_gen, "dna_id": best_org.dna_id})
            self.champion = best_org

        if best_fns is not None:
            brain.dna_plasticity_kernel, brain.dna_ironing_kernel = best_fns
        else:
            _, p_fn, i_fn, _ = self.compile_and_sandbox_dna(self.champion.source_code)
            brain.dna_plasticity_kernel, brain.dna_ironing_kernel = p_fn, i_fn

        self._prune_archive()
        if champion_changed:
            self.save_active_dna_to_disk(self.champion)
            self._append_lineage(self.champion, committed_genes)

        champ = self.champion
        return {
            "selected_dna_id": champ.dna_id,
            "champion_changed": champion_changed,
            "incumbent_fitness": None if incumbent_fitness is None else round(incumbent_fitness, 3),
            "ast_node_count": champ.ast_node_count,
            "active_p_genes": len(champ.p_gene_src),
            "active_i_genes": len(champ.i_gene_src),
            "gene_bank_size": self.gene_bank_size,
            "novel_genes_committed": committed_genes,
            "sandbox_failures": sandbox_failures,
            "fitness": round(best_score, 3),
            "dna_file_path": self.dna_file_path,
            "source_code": champ.source_code,
        }

    def _prune_archive(self) -> None:
        if len(self.clade_archive) <= self.max_archive:
            return
        keep = {id(self.clade_archive[0])}  # 원조 아메바 DNA는 영구 보존
        if self.champion is not None:
            keep.add(id(self.champion))
        ranked = sorted(
            self.clade_archive,
            key=lambda d: 0.6 * d.immediate_fitness + 0.4 * d.descendant_cmp_score,
            reverse=True,
        )
        survivors = [d for d in ranked if id(d) in keep]
        for d in ranked:
            if len(survivors) >= self.max_archive:
                break
            if id(d) not in keep:
                survivors.append(d)
        self.clade_archive = [self.clade_archive[0]] + [d for d in survivors if d is not self.clade_archive[0]]

    # ------------------------------------------------------------------ 디스크 기록
    def save_active_dna_to_disk(self, org: DNACodeOrganism) -> None:
        header = (
            f"# ==============================================================================\n"
            f"# AUTO-GENERATED BY OUROBOROS AST SELF-CODER | DNA ID: {org.dna_id}\n"
            f"# Generation: {org.generation} | AST Nodes: {org.ast_node_count} | Gene Bank: {self.gene_bank_size}\n"
            f"# ==============================================================================\n\n"
        )
        with open(self.dna_file_path, "w", encoding="utf-8") as f:
            f.write(header + org.source_code + "\n")

    def _append_lineage(self, org: DNACodeOrganism, committed_genes: List[str]) -> None:
        rec = {
            "dna_id": org.dna_id,
            "generation": org.generation,
            "parent_ids": org.parent_ids,
            "fitness": round(org.immediate_fitness, 4),
            "ast_nodes": org.ast_node_count,
            "gene_bank_size": self.gene_bank_size,
            "novel_genes_committed": committed_genes,
            "source_code": org.source_code,
        }
        with open(self.lineage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from OUROBOROS_01_energy_bedroom_and_spikes import UniversalBedroomFloor
    from OUROBOROS_02_amoeba_morphogenetic_brain import AmoebaMorphogeneticBrain
    from OUROBOROS_03_steam_iron_and_hydraulic_press import (
        CeilingHydraulicPressBank,
        SuperpositionSteamIronSolver,
    )

    floor = UniversalBedroomFloor(seed=42)
    brain = AmoebaMorphogeneticBrain(seed=42)
    bank = CeilingHydraulicPressBank()
    solver = SuperpositionSteamIronSolver(brain, bank, floor)
    coder = HuxleyGodelASTSelfCoder(output_dir=".")

    test_spikes = [floor.spawn_spike_problem(complexity_tier=2) for _ in range(3)]
    for g in range(1, 6):
        info = coder.evolve_and_hotswap_dna(brain, solver, test_spikes, next_gen=g)
        print(f"[Module 4 Verification] Gen {g}: {info['selected_dna_id']} "
              f"(AST Nodes: {info['ast_node_count']}, Gene Bank: {info['gene_bank_size']}, "
              f"New Genes: {len(info['novel_genes_committed'])}, Fitness: {info['fitness']})")
    print(coder.champion.source_code)
