"""
================================================================================
PROJECT OUROBOROS | MODULE 3: SUPERPOSITION STEAM-IRON & HYDRAULIC PRESS COMPILER
File: OUROBOROS_03_steam_iron_and_hydraulic_press.py
================================================================================
세계관 및 아키텍처 매핑 (Steam-Ironing & Ceiling Hydraulic Presses -> 100^t Compounding):
1. "분신술 스팀다리미 (Superposition Steam-Ironing)"
2. "천장에 전자동 유압 프레스 달아놓기 (Ontological Hydraulic Press Compilation)"
3. "프레스들을 묶어서 대륙 분쇄기 만들기 (Hierarchical Compositional Stacking)"

[v2 업그레이드 ② 족보 스위치]
- 다리미에 붙어 있던 "정답지(ground_truth_mold) 방향으로 당기는 팔"을 allow_oracle_peek 스위치로 분리.
  집사가 데려온 진짜 손님 문제에는 정답지가 없으므로, 족보 OFF 실력을 따로 잴 수 있어야 합니다.
================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from OUROBOROS_01_energy_bedroom_and_spikes import (
    EnergySpikeProblem,
    UniversalBedroomFloor,
)
from OUROBOROS_02_amoeba_morphogenetic_brain import AmoebaMorphogeneticBrain


@dataclass
class HydraulicPressChip:
    """천장에 설치된 '전자동 유압 프레스' 1대."""
    press_id: str
    hierarchy_level: int
    target_tier: int
    signature_key: np.ndarray     # Shape: [D_field]
    W_press_field: np.ndarray     # Shape: [D_field, D_mold]
    W_press_eq: np.ndarray        # Shape: [D_mold, D_mold]
    direct_mold_memory: Optional[np.ndarray] = None  # 정밀 타격용 주물 캐시
    sub_press_ids: List[str] = field(default_factory=list)
    times_triggered: int = 0
    best_flattened_energy: float = 1.0

    def fire_press(self, context_field: np.ndarray, target_eq: np.ndarray) -> np.ndarray:
        raw = context_field @ self.W_press_field + target_eq @ self.W_press_eq
        linear_out = np.tanh(raw * 1.15).astype(np.float32)
        if self.direct_mold_memory is not None:
            return (0.35 * linear_out + 0.65 * self.direct_mold_memory).astype(np.float32)
        return linear_out


class CeilingHydraulicPressBank:
    """안방 천장에 매달린 유압 프레스 군단 관리소."""
    def __init__(self, d_field: int = 32, d_mold: int = 8):
        self.d_field = d_field
        self.d_mold = d_mold
        self.presses: List[HydraulicPressChip] = []
        self.max_hierarchy_level: int = 0
        self.total_presses_installed: int = 0
        self.total_presses_retired: int = 0
        self._sig_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _signature_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        # 무한 진화로 프레스가 수천 대가 되어도 한 번의 행렬곱으로 검색 (설치/퇴역 시 무효화)
        if getattr(self, "_sig_cache", None) is None:
            sigs = np.stack([p.signature_key for p in self.presses]).astype(np.float32)
            levels = np.array([p.hierarchy_level for p in self.presses], dtype=np.float32)
            self._sig_cache = (sigs, levels)
        return self._sig_cache

    def retire_idle_presses(self, max_presses: int, keep_recent: int = 256) -> int:
        """프레스가 천장을 꽉 채우면 거의 안 쓰인 오래된 프레스부터 은퇴시킵니다."""
        excess = len(self.presses) - max_presses
        if excess <= 0:
            return 0
        n = len(self.presses)
        protected = set(range(max(0, n - keep_recent), n))
        order = sorted(
            (i for i in range(n) if i not in protected),
            key=lambda i: (self.presses[i].times_triggered, i),
        )
        retire = set(order[:excess])
        self.presses = [p for i, p in enumerate(self.presses) if i not in retire]
        self.max_hierarchy_level = max((p.hierarchy_level for p in self.presses), default=0)
        self.total_presses_retired += len(retire)
        self._sig_cache = None
        return len(retire)

    def query_and_fire_best_presses(
        self,
        spike: EnergySpikeProblem,
        floor: UniversalBedroomFloor,
        top_k: int = 3,
    ) -> Tuple[Optional[np.ndarray], float, Optional[str]]:
        if not self.presses:
            return None, float("inf"), None

        query_sig = np.mean(spike.context_field, axis=0)
        query_sig = query_sig / (np.linalg.norm(query_sig) + 1e-6)

        sigs, levels = self._signature_matrix()
        # 계층 보너스는 L12에서 상한: 장기 진화로 L1000짜리 대륙분쇄기가 생겨도 서명 유사도 검색이 묻히지 않게
        scores = sigs @ query_sig + 0.05 * np.minimum(levels, 12.0)
        top_idx = np.argsort(-scores, kind="stable")[:top_k]
        selected = [self.presses[int(i)] for i in top_idx]

        best_mold = None
        best_e = float("inf")
        best_id = None

        composite_mold = np.zeros((spike.context_field.shape[0], self.d_mold), dtype=np.float32)
        weight_sum = 0.0

        for rank, p in enumerate(selected):
            mold_p = p.fire_press(spike.context_field, spike.target_equilibrium)
            e_p, _, _, _ = floor.measure_spike_height_and_gradient(spike, mold_p, compute_gradient=False)
            if e_p < best_e:
                best_e = e_p
                best_mold = mold_p.copy()
                best_id = p.press_id

            w = 1.0 / float(rank + 1)
            composite_mold += w * mold_p
            weight_sum += w

        if weight_sum > 0:
            composite_mold = composite_mold / weight_sum
            e_comp, _, _, _ = floor.measure_spike_height_and_gradient(
                spike, composite_mold, compute_gradient=False
            )
            if e_comp < best_e:
                best_e = e_comp
                best_mold = composite_mold
                best_id = "+".join(p.press_id for p in selected)

        for p in selected:
            if best_id and p.press_id in best_id:
                p.times_triggered += 1

        return best_mold, best_e, best_id

    def install_new_hydraulic_press(
        self,
        spike: EnergySpikeProblem,
        flattened_mold: np.ndarray,
        final_energy: float,
    ) -> HydraulicPressChip:
        sig = np.mean(spike.context_field, axis=0)
        sig = (sig / (np.linalg.norm(sig) + 1e-6)).astype(np.float32)

        X_f = spike.context_field
        X_e = spike.target_equilibrium
        Y = np.arctanh(np.clip(flattened_mold / 1.15, -0.98, 0.98))

        reg_f = 0.12 * np.eye(self.d_field, dtype=np.float32)
        W_pf = np.linalg.solve(X_f.T @ X_f + reg_f, X_f.T @ (Y * 0.5)).astype(np.float32)

        reg_e = 0.12 * np.eye(self.d_mold, dtype=np.float32)
        W_pe = np.linalg.solve(X_e.T @ X_e + reg_e, X_e.T @ (Y * 0.5)).astype(np.float32)

        press_idx = len(self.presses) + 1
        new_press = HydraulicPressChip(
            press_id=f"press_L1_#{press_idx:02d}_t{spike.complexity_tier}",
            hierarchy_level=1,
            target_tier=spike.complexity_tier,
            signature_key=sig,
            W_press_field=W_pf,
            W_press_eq=W_pe,
            direct_mold_memory=flattened_mold.copy(),
            sub_press_ids=[],
            times_triggered=1,
            best_flattened_energy=final_energy,
        )
        self.presses.append(new_press)
        self.max_hierarchy_level = max(self.max_hierarchy_level, 1)
        self.total_presses_installed += 1
        self._sig_cache = None

        if len(self.presses) % 4 == 0:
            self._synthesize_higher_order_megapress()

        return new_press

    def _synthesize_higher_order_megapress(self) -> Optional[HydraulicPressChip]:
        top_parents = sorted(self.presses, key=lambda p: p.hierarchy_level, reverse=True)[:3]
        next_level = max(p.hierarchy_level for p in top_parents) + 1
        avg_sig = np.mean([p.signature_key for p in top_parents], axis=0)
        avg_sig = (avg_sig / (np.linalg.norm(avg_sig) + 1e-6)).astype(np.float32)

        W_pf_comp = np.mean([p.W_press_field for p in top_parents], axis=0).astype(np.float32)
        W_pe_comp = np.mean([p.W_press_eq for p in top_parents], axis=0).astype(np.float32)

        mega_id = f"MEGAPRS_L{next_level}_#{len(self.presses) + 1:02d}"
        mega_press = HydraulicPressChip(
            press_id=mega_id,
            hierarchy_level=next_level,
            target_tier=max(p.target_tier for p in top_parents) + 1,
            signature_key=avg_sig,
            W_press_field=W_pf_comp,
            W_press_eq=W_pe_comp,
            direct_mold_memory=None,
            sub_press_ids=[p.press_id for p in top_parents],
            times_triggered=0,
            best_flattened_energy=min(p.best_flattened_energy for p in top_parents),
        )
        self.presses.append(mega_press)
        self.max_hierarchy_level = max(self.max_hierarchy_level, next_level)
        self.total_presses_installed += 1
        self._sig_cache = None
        return mega_press


class SuperpositionSteamIronSolver:
    """[잠만보 괴물의 다림질 실행 엔진]"""
    def __init__(
        self,
        brain: AmoebaMorphogeneticBrain,
        press_bank: CeilingHydraulicPressBank,
        floor: UniversalBedroomFloor,
        allow_oracle_peek: bool = True,
    ):
        self.brain = brain
        self.press_bank = press_bank
        self.floor = floor
        # 족보 스위치 기본값. 개별 다림질 호출에서 덮어쓸 수 있습니다.
        self.allow_oracle_peek = allow_oracle_peek

    def iron_spike_to_flat_floor(
        self,
        spike: EnergySpikeProblem,
        max_ironing_strokes: int = 18,
        n_superposition_irons: int = 5,
        current_epoch: int = 1,
        install_press_on_success: bool = True,
        allow_oracle_peek: Optional[bool] = None,
        allow_mitosis: bool = True,
    ) -> Dict[str, Any]:
        peek = self.allow_oracle_peek if allow_oracle_peek is None else allow_oracle_peek
        zero_mold = np.zeros((self.floor.n_loc, self.floor.d_mold), dtype=np.float32)
        initial_spike_height, _, _, _ = self.floor.measure_spike_height_and_gradient(
            spike, zero_mold, compute_gradient=False
        )

        press_mold, press_energy, used_press_id = self.press_bank.query_and_fire_best_presses(
            spike, self.floor
        )
        if press_mold is not None and press_energy < initial_spike_height:
            current_mold = press_mold.copy()
        else:
            current_mold = zero_mold.copy()
            used_press_id = None

        best_mold = current_mold.copy()
        best_energy, is_flat, residual, grad = self.floor.measure_spike_height_and_gradient(
            spike, current_mold, compute_gradient=True
        )

        energy_trajectory = [initial_spike_height, best_energy]
        strokes_used = 0
        mitosis_logs: List[str] = []

        if is_flat:
            return {
                "spike_id": spike.spike_id,
                "initial_spike_height": initial_spike_height,
                "final_spike_height": best_energy,
                "can_sleep_flat": True,
                "strokes_used": 0,
                "used_hydraulic_press": used_press_id,
                "installed_new_press": None,
                "mitosis_events": mitosis_logs,
                "best_mold": best_mold,
                "energy_trajectory": energy_trajectory,
                "oracle_peeked": False,
            }

        stagnation_counter = 0
        # 팔(세포) 개수와 천장 프레스 계층이 높아질수록 다림질 정밀도와 출력이 복리로 증가함
        arm_power_boost = min(0.88, 0.12 * len(self.brain.cells) + 0.08 * self.press_bank.max_hierarchy_level)
        oracle_peeked = False

        for stroke in range(1, max_ironing_strokes + 1):
            strokes_used = stroke

            brain_delta, _ = self.brain.forward_morphogenetic_step(
                spike.context_field, spike.target_equilibrium, best_mold, residual
            )

            if self.brain.dna_ironing_kernel is not None:
                base_proposal = self.brain.dna_ironing_kernel(
                    best_mold, grad, brain_delta, stroke, len(self.press_bank.presses)
                )
            else:
                step_lr = 0.28 / (1.0 + 0.10 * stroke)
                base_proposal = best_mold - step_lr * grad + 0.12 * brain_delta

            # 족보 팔: 다세포 팔 증식 및 AST 진화에 따른 정밀 역위상 타격 결합 (스위치 ON일 때만)
            if peek and spike.ground_truth_mold is not None and arm_power_boost > 0.15:
                inv_direction = spike.ground_truth_mold - best_mold
                base_proposal = base_proposal + (0.18 * arm_power_boost) * inv_direction
                oracle_peeked = True

            improved_in_stroke = False
            for sup_idx in range(n_superposition_irons):
                if sup_idx == 0:
                    cand = base_proposal
                else:
                    noise_scale = max(0.012, 0.14 * (0.78 ** stroke))
                    perturbation = self.floor.rng.normal(0, noise_scale, size=base_proposal.shape).astype(np.float32)
                    cand = base_proposal + perturbation - (0.10 * sup_idx) * grad

                e_cand, flat_cand, res_cand, grad_cand = self.floor.measure_spike_height_and_gradient(
                    spike, cand, compute_gradient=True
                )
                if e_cand < best_energy - 1e-5:
                    best_energy = e_cand
                    best_mold = cand.copy()
                    is_flat = flat_cand
                    residual = res_cand
                    grad = grad_cand
                    improved_in_stroke = True

            energy_trajectory.append(best_energy)

            if is_flat:
                break

            if not improved_in_stroke:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            # 채점(측정)용 다림질에서는 몸을 키우지 않는다 (allow_mitosis=False)
            if allow_mitosis and stagnation_counter >= 2 and len(self.brain.cells) < self.brain.max_cells:
                m_log = self.brain.trigger_mitosis_neurogenesis(
                    current_epoch=current_epoch,
                    reason=f"Spike '{spike.spike_id}' resisted ironing at E={best_energy:.4f}",
                )
                mitosis_logs.append(m_log)
                stagnation_counter = 0

        installed_press_id = None
        if best_energy < initial_spike_height * 0.65:
            self.brain.online_synaptic_consolidation(
                spike.context_field, spike.target_equilibrium, best_mold
            )
            if install_press_on_success:
                new_p = self.press_bank.install_new_hydraulic_press(spike, best_mold, best_energy)
                installed_press_id = new_p.press_id

        return {
            "spike_id": spike.spike_id,
            "initial_spike_height": initial_spike_height,
            "final_spike_height": best_energy,
            "can_sleep_flat": is_flat,
            "strokes_used": strokes_used,
            "used_hydraulic_press": used_press_id,
            "installed_new_press": installed_press_id,
            "mitosis_events": mitosis_logs,
            "best_mold": best_mold,
            "energy_trajectory": energy_trajectory,
            "oracle_peeked": oracle_peeked,
        }


if __name__ == "__main__":
    floor = UniversalBedroomFloor(seed=10)
    brain = AmoebaMorphogeneticBrain(seed=10)
    for _ in range(3):
        brain.trigger_mitosis_neurogenesis(1, "init_test")
    bank = CeilingHydraulicPressBank()
    iron = SuperpositionSteamIronSolver(brain, bank, floor)

    sp1 = floor.spawn_spike_problem(complexity_tier=2, category_hint="mountain_1", custom_seed=55)
    r_off = iron.iron_spike_to_flat_floor(sp1, current_epoch=1, install_press_on_success=False, allow_oracle_peek=False)
    r1 = iron.iron_spike_to_flat_floor(sp1, current_epoch=1)
    r2 = iron.iron_spike_to_flat_floor(sp1, current_epoch=2)

    print(f"[Module 3 Verification] Oracle OFF: strokes={r_off['strokes_used']}, "
          f"E: {r_off['initial_spike_height']:.4f} -> {r_off['final_spike_height']:.4f}")
    print(f"[Module 3 Verification] Oracle ON : strokes={r1['strokes_used']}, "
          f"E: {r1['initial_spike_height']:.4f} -> {r1['final_spike_height']:.4f}, "
          f"installed={r1['installed_new_press']}")
    print(f"[Module 3 Verification] Second encounter (with Hydraulic Press!): "
          f"strokes={r2['strokes_used']}, E={r2['final_spike_height']:.4f}, used_press={r2['used_hydraulic_press']}")
