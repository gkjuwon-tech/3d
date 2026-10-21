"""
================================================================================
PROJECT OUROBOROS | MODULE 1: THE BEDROOM FLOOR & UNIVERSAL SPIKE LANDSCAPE
File: OUROBOROS_01_energy_bedroom_and_spikes.py
================================================================================
세계관 및 아키텍처 매핑 (The Angry Sleeping Monster Metaphor -> Math & Code):
- "평평한 방바닥 (Energy = 0)": 모순이 전혀 없는 완벽한 평형 상태(Equilibrium).
  잠만보 괴물(우리의 <0.1B 코어)은 바닥이 평평할 때만 연산을 멈추고(Adaptive Halting) 잠을 잡니다.
- "에베레스트 가시 산 (Energy Spike E >> 0)":
  사용자의 질문, 수학 방정식, 신약 분자 결합 조건, 알고리즘 버그, 혹은 괴물 자신이 자다가
  불안해서 만들어낸 악몽(Self-Play Conjecture)은 모두 도메인 구분 없이
  안방 바닥에서 솟아오른 '다차원 가시 지형(Contradiction Energy Landscape)'으로 투영됩니다.
- "다림질 완료 후 찍혀 나온 주물 자국 (Ironed Mold y*)":
  가시 산을 해발 0미터(E -> 0)로 완벽하게 짓눌러 평평하게 만들었을 때 다리미 바닥에 남은
  주물 형태가 곧 우리가 찾는 수학 증명, 신약 분자 구조, 파이썬 코드, 또는 질문의 정답입니다.
================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class EnergySpikeProblem:
    """
    안방 바닥에 솟아오른 '가시 산(모순 지형)' 하나를 정의하는 보편 문제 객체.
    도메인용 별도 모듈을 두지 않고, 세상의 모든 문제를 동일한 구조로 통일합니다.
    """
    spike_id: str
    category_hint: str
    complexity_tier: int
    context_field: np.ndarray         # Shape: [N_loc, D_field]
    target_equilibrium: np.ndarray    # Shape: [N_loc, D_mold]
    constraint_operator: Dict[str, Any] = field(default_factory=dict)
    ground_truth_mold: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalBedroomFloor:
    """
    [안방 바닥 물리 엔진 (Universal Ground-Truth Energy Verifier)]
    - 잠만보가 들이민 다리미 주물(candidate_mold)이 가시 산의 비선형 제약 조건들을
      실제로 상쇄하여 방바닥 높이를 0(평지)으로 만들었는지를 엄격하게 측정합니다.
    """
    def __init__(self, n_loc: int = 16, d_field: int = 32, d_mold: int = 8, seed: int = 2026):
        self.n_loc = n_loc
        self.d_field = d_field
        self.d_mold = d_mold
        self.rng = np.random.default_rng(seed)
        self.flat_tolerance = 0.0145  # 이 높이 이하로 다림질되면 "평평함(숙면 가능)"으로 판정

    def forward_reality_physics(
        self,
        context_field: np.ndarray,
        candidate_mold: np.ndarray,
        ops: Dict[str, Any],
    ) -> np.ndarray:
        W_proj = ops["W_field_to_mold"]        # [D_field, D_mold]
        W_interact = ops["W_nonlinear_core"]   # [D_mold, D_mold]
        W_spatial = ops["W_spatial_coupling"]  # [N_loc, N_loc]
        freq_mod = ops["freq_mod"]             # float
        depth_steps = ops["depth_steps"]       # int

        state = np.tanh(context_field @ W_proj + candidate_mold)

        for _ in range(depth_steps):
            spatial_flow = W_spatial @ state
            nonlinear_term = np.sin(state * freq_mod) @ W_interact
            gate = 1.0 / (1.0 + np.exp(-(spatial_flow * 1.5)))
            state = np.tanh(0.55 * state + 0.45 * (gate * nonlinear_term + (1.0 - gate) * spatial_flow))

        return state

    def measure_spike_height_and_gradient(
        self,
        spike: EnergySpikeProblem,
        candidate_mold: np.ndarray,
        compute_gradient: bool = True,
    ) -> Tuple[float, bool, np.ndarray, np.ndarray]:
        mold_clipped = np.clip(candidate_mold, -2.5, 2.5).astype(np.float32)
        resulting_state = self.forward_reality_physics(
            spike.context_field, mold_clipped, spike.constraint_operator
        )
        residual = resulting_state - spike.target_equilibrium

        spike_energy = float(np.mean(residual ** 2))
        is_flat = bool(spike_energy <= self.flat_tolerance)

        if not compute_gradient:
            return spike_energy, is_flat, residual, np.zeros_like(mold_clipped)

        W_interact = spike.constraint_operator["W_nonlinear_core"]
        W_spatial = spike.constraint_operator["W_spatial_coupling"]
        depth_steps = spike.constraint_operator["depth_steps"]

        grad = residual.copy()
        for _ in range(min(depth_steps, 3)):
            grad = 0.6 * grad + 0.4 * (W_spatial.T @ grad @ W_interact.T)
        grad = grad / (np.linalg.norm(grad) + 1e-6)
        return spike_energy, is_flat, residual, grad.astype(np.float32)

    def spawn_spike_problem(
        self,
        complexity_tier: int = 1,
        category_hint: str = "universal_anomaly",
        custom_seed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EnergySpikeProblem:
        rng = np.random.default_rng(custom_seed) if custom_seed is not None else self.rng
        scale = 0.35 + 0.07 * min(complexity_tier, 12)
        depth_steps = min(1 + (complexity_tier + 1) // 2, 7)
        freq_mod = float(1.1 + 0.18 * min(complexity_tier, 10))

        W_field_to_mold = rng.normal(0, scale, size=(self.d_field, self.d_mold)).astype(np.float32)
        W_nonlinear = rng.normal(0, scale, size=(self.d_mold, self.d_mold)).astype(np.float32)
        W_nonlinear = 0.5 * (W_nonlinear - W_nonlinear.T) + 0.35 * np.eye(self.d_mold, dtype=np.float32)

        raw_spatial = rng.normal(0, 0.25, size=(self.n_loc, self.n_loc)).astype(np.float32)
        W_spatial = 0.65 * np.eye(self.n_loc, dtype=np.float32) + 0.35 * np.tanh(raw_spatial + raw_spatial.T)

        ops = {
            "W_field_to_mold": W_field_to_mold,
            "W_nonlinear_core": W_nonlinear,
            "W_spatial_coupling": W_spatial,
            "freq_mod": freq_mod,
            "depth_steps": depth_steps,
        }

        context_field = rng.uniform(-0.9, 0.9, size=(self.n_loc, self.d_field)).astype(np.float32)
        true_mold = rng.uniform(-1.1, 1.1, size=(self.n_loc, self.d_mold)).astype(np.float32)
        if complexity_tier >= 3:
            harmonic = np.sin(np.linspace(0, math.pi * 2, self.n_loc))[:, None]
            true_mold = np.tanh(true_mold + 0.4 * harmonic).astype(np.float32)

        target_eq = self.forward_reality_physics(context_field, true_mold, ops)

        return EnergySpikeProblem(
            spike_id=f"spike_{category_hint}_t{complexity_tier}_{int(rng.integers(10000, 99999))}",
            category_hint=category_hint,
            complexity_tier=complexity_tier,
            context_field=context_field,
            target_equilibrium=target_eq,
            constraint_operator=ops,
            ground_truth_mold=true_mold,
            metadata=metadata or {},
        )


if __name__ == "__main__":
    floor = UniversalBedroomFloor(seed=42)
    spike = floor.spawn_spike_problem(complexity_tier=3, category_hint="everest_test")
    e_raw, flat_raw, _, _ = floor.measure_spike_height_and_gradient(
        spike, np.zeros_like(spike.ground_truth_mold)
    )
    e_gt, flat_gt, _, _ = floor.measure_spike_height_and_gradient(
        spike, spike.ground_truth_mold
    )
    print(f"[Module 1 Verification] Spike '{spike.spike_id}':")
    print(f"  Un-ironed Spike Height (Ouch!): E = {e_raw:.5f} (Can sleep? {flat_raw})")
    print(f"  Perfectly Ironed Floor Height : E = {e_gt:.6f} (Can sleep? {flat_gt})")
