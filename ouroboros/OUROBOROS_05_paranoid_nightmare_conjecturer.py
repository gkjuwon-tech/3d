"""
================================================================================
PROJECT OUROBOROS | MODULE 5: THE PARANOID SLEEPER (NIGHTMARE SELF-PLAY ENGINE)
File: OUROBOROS_05_paranoid_nightmare_conjecturer.py
================================================================================
세계관 및 아키텍처 매핑 (Paranoid Sleeper That Spawns Nightmares When Idle):
- 손님이 아무 질문도 안 할 때 스스로 안방 바닥을 내리쳐서 '현재 천장의 유압 프레스들이
  가장 못 깎는 사각지대(Epistemic Blind Spot)의 악몽 가시 산'을 일부러 솟구치게 만든 뒤,
  혼자 씩씩대며 다림질하고 새 유압 프레스를 달아놓습니다!

[v2 업그레이드 ④ 악몽 × 셀프코더 콤보]
- 사냥한 사각지대 악몽들을 nightmare_archive에 쌓아두고, AST 셀프코더의 적합도 평가셋으로
  넘겨줍니다. 꿈에서 맞은 곳이 곧 다음 세대 유전자의 선택압이 됩니다.
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from OUROBOROS_01_energy_bedroom_and_spikes import (
    EnergySpikeProblem,
    UniversalBedroomFloor,
)
from OUROBOROS_03_steam_iron_and_hydraulic_press import (
    CeilingHydraulicPressBank,
    SuperpositionSteamIronSolver,
)


@dataclass
class NightmareHuntReport:
    epoch: int
    current_tier: int
    probed_candidates: int
    worst_blindspot_energy_before: float
    flattened_energy_after: float
    can_sleep_now: bool
    spawned_press_id: Optional[str]
    mitosis_triggered: List[str]
    spike_id: str = ""


class ParanoidNightmareConjecturer:
    """[잠꼬대 악몽 생성기 (Autonomous Epistemic Anomaly Hunter)]"""
    def __init__(self, floor: UniversalBedroomFloor, seed: int = 2026, archive_size: int = 8):
        self.floor = floor
        self.rng = np.random.default_rng(seed)
        self.current_nightmare_tier: int = 1
        self.consecutive_flat_wins: int = 0
        self.archive_size = archive_size
        self.nightmare_archive: List[EnergySpikeProblem] = []
        self.total_nightmares_hunted: int = 0

    def recent_nightmares(self, k: int) -> List[EnergySpikeProblem]:
        """가장 최근에 사냥한 사각지대 악몽 k개 (셀프코더 평가셋용)."""
        return list(self.nightmare_archive[-k:])

    def hunt_and_conquer_nightmare(
        self,
        epoch: int,
        solver: SuperpositionSteamIronSolver,
        press_bank: CeilingHydraulicPressBank,
        n_nightmare_probes: int = 4,
    ) -> NightmareHuntReport:
        worst_spike: Optional[EnergySpikeProblem] = None
        worst_press_energy = -1.0

        for _ in range(n_nightmare_probes):
            cand_spike = self.floor.spawn_spike_problem(
                complexity_tier=self.current_nightmare_tier,
                category_hint="nightmare_selfplay",
            )
            _, press_e, _ = press_bank.query_and_fire_best_presses(cand_spike, self.floor)
            if not np.isfinite(press_e):
                zero_m = np.zeros((self.floor.n_loc, self.floor.d_mold), dtype=np.float32)
                press_e, _, _, _ = self.floor.measure_spike_height_and_gradient(
                    cand_spike, zero_m, compute_gradient=False
                )
            if press_e > worst_press_energy:
                worst_press_energy = press_e
                worst_spike = cand_spike

        assert worst_spike is not None
        self.nightmare_archive.append(worst_spike)
        if len(self.nightmare_archive) > self.archive_size:
            self.nightmare_archive = self.nightmare_archive[-self.archive_size:]
        self.total_nightmares_hunted += 1

        iron_res = solver.iron_spike_to_flat_floor(
            worst_spike,
            max_ironing_strokes=16,
            n_superposition_irons=5,
            current_epoch=epoch,
            install_press_on_success=True,
        )

        if iron_res["can_sleep_flat"]:
            self.consecutive_flat_wins += 1
            if self.consecutive_flat_wins >= 2 and self.current_nightmare_tier < 12:
                self.current_nightmare_tier += 1
                self.consecutive_flat_wins = 0
        else:
            self.consecutive_flat_wins = 0

        return NightmareHuntReport(
            epoch=epoch,
            current_tier=worst_spike.complexity_tier,
            probed_candidates=n_nightmare_probes,
            worst_blindspot_energy_before=iron_res["initial_spike_height"],
            flattened_energy_after=iron_res["final_spike_height"],
            can_sleep_now=iron_res["can_sleep_flat"],
            spawned_press_id=iron_res["installed_new_press"],
            mitosis_triggered=iron_res["mitosis_events"],
            spike_id=worst_spike.spike_id,
        )


if __name__ == "__main__":
    from OUROBOROS_02_amoeba_morphogenetic_brain import AmoebaMorphogeneticBrain

    floor = UniversalBedroomFloor(seed=99)
    brain = AmoebaMorphogeneticBrain(seed=99)
    bank = CeilingHydraulicPressBank()
    solver = SuperpositionSteamIronSolver(brain, bank, floor)
    nightmare = ParanoidNightmareConjecturer(floor, seed=99)

    for ep in range(1, 5):
        rep = nightmare.hunt_and_conquer_nightmare(ep, solver, bank)
        print(f"[Module 5 Verification | Nightmare Ep {ep}] Tier={rep.current_tier} | "
              f"Spike: {rep.worst_blindspot_energy_before:.4f} -> {rep.flattened_energy_after:.4f} | "
              f"Sleep={rep.can_sleep_now} | NewPress={rep.spawned_press_id}")
    print(f"  Nightmare archive for self-coder: {[s.spike_id for s in nightmare.recent_nightmares(4)]}")
