"""
================================================================================
PROJECT OUROBOROS | MODULE 2: THE AMOEBA THAT GROWS ARMS (MORPHOGENETIC BRAIN)
File: OUROBOROS_02_amoeba_morphogenetic_brain.py
================================================================================
세계관 및 아키텍처 매핑 (Amoeba -> Morphogenetic Multi-Cellular Brain):
- "처음에는 아메바":
  태어날 때는 단 1개의 작은 연산 세포(Operator Cell, 히든 차원 24)만 가진 진짜 '아메바'로 출발합니다.
- "바위가 안 깨지면 어깨를 찢고 팔을 4개, 8개로 세포 분열시킨다 (Neurogenesis / Mitosis)":
  가시 산을 다림질하다가 현재 뇌세포들의 힘만으로는 도저히 모순 에너지가 안 줄어드는 병목이 감지되면,
  가장 과부하가 걸린 뇌세포가 스스로 두 개로 세포 분열(Mitosis)하며 새로운 비선형 신경 통로(팔)를 증설합니다!
================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class MorphogeneticCell:
    """살아있는 형태발생 뇌를 구성하는 개별 '연산 세포(팔/근육 다발)' 하나."""
    cell_id: str
    birth_epoch: int
    parent_cell_id: Optional[str]
    d_in: int
    d_cell: int
    d_out: int
    act_mode: str  # ["tanh", "sin_wave", "cubic_hammer", "swish_blade"]
    W_enc: np.ndarray     # [d_in, d_cell]
    W_rec: np.ndarray     # [d_cell, d_cell]
    W_dec: np.ndarray     # [d_cell, d_out]
    W_fast: np.ndarray    # [d_cell, d_cell] 런타임에 실시간으로 변하는 자기 참조 시냅스(SRWM)
    stress_accumulator: float = 0.0
    utility_score: float = 1.0

    def parameter_count(self) -> int:
        return int(self.W_enc.size + self.W_rec.size + self.W_dec.size + self.W_fast.size)


class AmoebaMorphogeneticBrain:
    """
    [단세포 아메바에서 시작해 스스로 세포 분열·합체하며 초지능으로 자라나는 액체 신경망]
    """
    def __init__(
        self,
        n_loc: int = 16,
        d_field: int = 32,
        d_mold: int = 8,
        initial_cell_dim: int = 24,
        max_cells: int = 32,
        seed: int = 2026,
    ):
        self.n_loc = n_loc
        self.d_field = d_field
        self.d_mold = d_mold
        self.d_input_total = d_field + d_mold * 2
        self.initial_cell_dim = initial_cell_dim
        self.max_cells = max_cells
        self.rng = np.random.default_rng(seed)

        self.cells: List[MorphogeneticCell] = []
        self.total_mitosis_events: int = 0
        self.total_apoptosis_events: int = 0

        self.dna_plasticity_kernel: Optional[Callable] = None
        self.dna_ironing_kernel: Optional[Callable] = None

        self._spawn_primordial_amoeba()

    def _spawn_primordial_amoeba(self) -> None:
        d_c = self.initial_cell_dim
        scale_in = 1.0 / math.sqrt(self.d_input_total)
        scale_c = 1.0 / math.sqrt(d_c)

        amoeba_cell = MorphogeneticCell(
            cell_id="cell_0_primordial_amoeba",
            birth_epoch=0,
            parent_cell_id=None,
            d_in=self.d_input_total,
            d_cell=d_c,
            d_out=self.d_mold,
            act_mode="tanh",
            W_enc=self.rng.normal(0, scale_in, (self.d_input_total, d_c)).astype(np.float32),
            W_rec=(np.eye(d_c, dtype=np.float32) * 0.5 +
                   self.rng.normal(0, scale_c * 0.3, (d_c, d_c)).astype(np.float32)),
            W_dec=self.rng.normal(0, scale_c, (d_c, self.d_mold)).astype(np.float32),
            W_fast=(np.eye(d_c, dtype=np.float32) * 0.15),
            stress_accumulator=0.0,
            utility_score=1.0,
        )
        self.cells.append(amoeba_cell)

    def _apply_cell_activation(self, x: np.ndarray, mode: str) -> np.ndarray:
        if mode == "sin_wave":
            return np.sin(x * 1.35).astype(np.float32)
        elif mode == "cubic_hammer":
            return np.tanh(x + 0.45 * (x ** 3)).astype(np.float32)
        elif mode == "swish_blade":
            sig = 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))
            return np.tanh(x * sig * 1.4).astype(np.float32)
        return np.tanh(x).astype(np.float32)

    def forward_morphogenetic_step(
        self,
        context_field: np.ndarray,
        target_eq: np.ndarray,
        current_mold: np.ndarray,
        residual_spike: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        inp = np.concatenate([context_field, target_eq, current_mold], axis=-1).astype(np.float32)
        total_delta_mold = np.zeros_like(current_mold, dtype=np.float32)
        spike_intensity = float(np.mean(residual_spike ** 2))

        for cell in self.cells:
            h_raw = inp @ cell.W_enc
            h_rec = h_raw @ (cell.W_rec + cell.W_fast)
            h_act = self._apply_cell_activation(h_rec, cell.act_mode)

            cell_force = h_act @ cell.W_dec
            weight_share = 1.0 / math.sqrt(len(self.cells))
            total_delta_mold = total_delta_mold + weight_share * cell_force

            alignment = float(np.mean(-cell_force * residual_spike))
            cell.stress_accumulator = 0.85 * cell.stress_accumulator + 0.15 * (
                spike_intensity * float(np.std(h_act))
            )
            cell.utility_score = 0.90 * cell.utility_score + 0.10 * (abs(alignment) + 0.05)

            k_vec = np.mean(h_raw, axis=0)
            k_vec = k_vec / (np.linalg.norm(k_vec) + 1e-6)
            v_vec = np.mean(h_act, axis=0)

            if self.dna_plasticity_kernel is not None:
                cell.W_fast = self.dna_plasticity_kernel(cell.W_fast, k_vec, v_vec, spike_intensity)
            else:
                wk = cell.W_fast @ k_vec
                err = v_vec - wk
                delta_fast = 0.18 * np.outer(err, k_vec)
                cell.W_fast = np.clip(0.92 * cell.W_fast + delta_fast, -1.5, 1.5).astype(np.float32)

        return np.tanh(total_delta_mold).astype(np.float32), spike_intensity

    def trigger_mitosis_neurogenesis(self, current_epoch: int, reason: str = "high_spike_stress") -> str:
        if len(self.cells) >= self.max_cells:
            self.trigger_apoptosis_pruning()

        stressed_parent = max(self.cells, key=lambda c: c.stress_accumulator)
        stressed_parent.stress_accumulator *= 0.4

        new_d_cell = min(stressed_parent.d_cell + 8, 96)
        modes = ["tanh", "sin_wave", "cubic_hammer", "swish_blade"]
        new_mode = str(self.rng.choice([m for m in modes if m != stressed_parent.act_mode] or modes))

        scale_in = 1.0 / math.sqrt(self.d_input_total)
        scale_c = 1.0 / math.sqrt(new_d_cell)

        W_enc_new = self.rng.normal(0, scale_in, (self.d_input_total, new_d_cell)).astype(np.float32)
        old_d = stressed_parent.d_cell
        W_enc_new[:, :old_d] = stressed_parent.W_enc + self.rng.normal(0, 0.03, stressed_parent.W_enc.shape).astype(np.float32)

        W_rec_new = np.eye(new_d_cell, dtype=np.float32) * 0.45 + self.rng.normal(0, scale_c * 0.25, (new_d_cell, new_d_cell)).astype(np.float32)
        W_rec_new[:old_d, :old_d] = stressed_parent.W_rec

        W_dec_new = self.rng.normal(0, scale_c, (new_d_cell, self.d_mold)).astype(np.float32)
        W_dec_new[:old_d, :] = stressed_parent.W_dec * 0.8

        self.total_mitosis_events += 1
        child_id = f"cell_{self.total_mitosis_events}_{new_mode}_{new_d_cell}d"
        child_cell = MorphogeneticCell(
            cell_id=child_id,
            birth_epoch=current_epoch,
            parent_cell_id=stressed_parent.cell_id,
            d_in=self.d_input_total,
            d_cell=new_d_cell,
            d_out=self.d_mold,
            act_mode=new_mode,
            W_enc=W_enc_new,
            W_rec=W_rec_new,
            W_dec=W_dec_new,
            W_fast=np.eye(new_d_cell, dtype=np.float32) * 0.15,
            stress_accumulator=0.1,
            utility_score=stressed_parent.utility_score,
        )
        self.cells.append(child_cell)
        return (
            f"[MITOSIS! 팔 증설] 부모 '{stressed_parent.cell_id}' 분열 -> "
            f"신규 세포 '{child_id}' 탄생! (현재 팔 개수: {len(self.cells)}개, 사유: {reason})"
        )

    def trigger_apoptosis_pruning(self) -> Optional[str]:
        if len(self.cells) <= 2:
            return None
        weakest = min(self.cells[1:], key=lambda c: c.utility_score)
        self.cells.remove(weakest)
        self.total_apoptosis_events += 1
        return f"[APOPTOSIS 가지치기] 저효율 세포 '{weakest.cell_id}' 소멸 (남은 세포: {len(self.cells)}개)"

    def online_synaptic_consolidation(
        self,
        context_field: np.ndarray,
        target_eq: np.ndarray,
        successful_mold: np.ndarray,
        lr: float = 0.04,
    ) -> None:
        inp = np.concatenate([context_field, target_eq, np.zeros_like(successful_mold)], axis=-1).astype(np.float32)
        for cell in self.cells:
            h_raw = inp @ cell.W_enc
            h_act = self._apply_cell_activation(h_raw @ (cell.W_rec + cell.W_fast), cell.act_mode)
            pred = h_act @ cell.W_dec
            err = pred - successful_mold
            grad_dec = (h_act.T @ err) / float(self.n_loc)
            cell.W_dec = np.clip(cell.W_dec - lr * grad_dec, -2.5, 2.5).astype(np.float32)

    def get_brain_vital_signs(self) -> Dict[str, Any]:
        total_params = sum(c.parameter_count() for c in self.cells)
        return {
            "num_cells_arms": len(self.cells),
            "total_parameters": total_params,
            "under_0_1B_limit": total_params < 100_000_000,
            "mitosis_count": self.total_mitosis_events,
            "apoptosis_count": self.total_apoptosis_events,
            "cell_lineages": [f"{c.cell_id}({c.d_cell}d)" for c in self.cells],
        }


if __name__ == "__main__":
    brain = AmoebaMorphogeneticBrain()
    print("[Module 2 Verification] Stage 0 Amoeba Vital Signs:", brain.get_brain_vital_signs())
    msg1 = brain.trigger_mitosis_neurogenesis(current_epoch=1, reason="everest_spike_detected")
    msg2 = brain.trigger_mitosis_neurogenesis(current_epoch=2, reason="diamond_rock_detected")
    print(" ", msg1)
    print(" ", msg2)
    print("[Module 2 Verification] After Mitosis Vital Signs:", brain.get_brain_vital_signs())
