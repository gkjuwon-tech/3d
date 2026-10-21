"""
================================================================================
PROJECT OUROBOROS | MODULE 7: KAGGLE INFINITE EVOLUTION DAEMON (AMOEBA -> ASI)
File: OUROBOROS_07_kaggle_infinite_daemon.py
================================================================================
세계관 및 아키텍처 매핑 (Continuous Kaggle Evolution from Single-Cell Amoeba to ASI):
- Kaggle에서 켜놓고 계속 돌릴 수 있는 무한 자율 진화 데몬(Daemon)입니다.

[v2 업그레이드]
- ② 족보 ON/OFF 이중 채점 + 처음 보는 손님(unseen walk-in) 시험까지 3중 계기판
- ③ 체크포인트 이원화(.pkl / .pkl.bak 번갈아 원자적 저장), rng 상태 통째 보존,
     /kaggle/input 에 이전 세션 체크포인트가 있으면 자동으로 이어받기
- ④ 악몽 아카이브를 AST 셀프코더 평가셋으로 투입
- 통역사(집사) 상시 근무: 매 에포크 처음 보는 손님 요청을 받아 다림질하고 브리핑을 남김
- 시간 예산(--hours): Kaggle 12시간 제한 전에 스스로 마무리하고 결과물을 저장
================================================================================
"""

from __future__ import annotations
import argparse
import glob
import json
import math
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from OUROBOROS_01_energy_bedroom_and_spikes import (
    EnergySpikeProblem,
    UniversalBedroomFloor,
)
from OUROBOROS_02_amoeba_morphogenetic_brain import AmoebaMorphogeneticBrain
from OUROBOROS_03_steam_iron_and_hydraulic_press import (
    CeilingHydraulicPressBank,
    SuperpositionSteamIronSolver,
)
from OUROBOROS_04_ast_dna_self_coder import HuxleyGodelASTSelfCoder
from OUROBOROS_05_paranoid_nightmare_conjecturer import ParanoidNightmareConjecturer
from OUROBOROS_06_butler_transducer_and_decoders import PoliteButlerTransducer


CHECKPOINT_NAME = "ouroboros_infinite_checkpoint.pkl"


class OuroborosInfiniteEvolutionDaemon:
    def __init__(
        self,
        output_dir: str = ".",
        seed: int = 2026,
        max_presses: int = 6000,
        guests_per_epoch: int = 2,
        unseen_guests_per_eval: int = 8,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(output_dir, CHECKPOINT_NAME)
        self.telemetry_path = os.path.join(output_dir, "ouroboros_evolution_telemetry.json")
        self.chart_path = os.path.join(output_dir, "ouroboros_amoeba_to_asi_proof.png")
        self.guest_log_path = os.path.join(output_dir, "butler_guest_log.jsonl")

        self.max_presses = max_presses
        self.guests_per_epoch = guests_per_epoch
        self.unseen_guests_per_eval = unseen_guests_per_eval

        self.floor = UniversalBedroomFloor(seed=seed)
        self.brain = AmoebaMorphogeneticBrain(seed=seed)
        self.press_bank = CeilingHydraulicPressBank()
        self.solver = SuperpositionSteamIronSolver(self.brain, self.press_bank, self.floor)
        self.ast_coder = HuxleyGodelASTSelfCoder(output_dir=output_dir, seed=seed)
        self.nightmare = ParanoidNightmareConjecturer(self.floor, seed=seed)
        self.butler = PoliteButlerTransducer(self.floor)
        self.guest_rng = np.random.default_rng(seed + 7)

        self.current_epoch: int = 0
        self.active_dna_id: str = "dna_gen0_amoeba_seed"
        self.history: List[Dict[str, Any]] = []
        self.recent_briefings: List[Dict[str, Any]] = []
        self.sessions: List[Dict[str, Any]] = []

        self.everest_benchmark_spikes = self._build_4domain_everest_suite()

    def _build_4domain_everest_suite(self) -> List[EnergySpikeProblem]:
        prompts = [
            ("KRAS G12D 변이 암 단백질을 간 독성 없이 억제하는 신약 분자 설계", "drug_molecular"),
            ("EGFR 삼중 변이 내성 표적을 돌파하는 고투과성 리간드 역설계", "drug_molecular"),
            ("다중 키나아제 표적 선택성 극대화 및 심장 독성(hERG) 0 수렴 분자 합성", "drug_molecular"),
            ("비선형 결합 편미분 경계조건 하에서의 최소 작용 고유상태 증명", "math_symbolic"),
            ("고차 다항 행렬 방정식의 비자명 불변 근(Invariant Root) 역산", "math_symbolic"),
            ("비대칭 위상 그래프 상의 전역 에너지 평형 정리 유도", "math_symbolic"),
            ("O(1) 메모리 복잡도를 갖는 비선형 상태 전이 커널 코드 합성", "code_program"),
            ("자기 참조 텐서 압축 및 병목 제거 최적화 알고리즘 생성", "code_program"),
            ("다중 제약 충돌을 실시간 중재하는 비동기 스케줄러 함수 합성", "code_program"),
            ("에너지 비용을 절반으로 줄이면서 추론 깊이를 제곱으로 늘리는 전략 도출", "dialogue_logic"),
            ("한정된 GPU 메모리에서 치명적 망각 없이 무한 학습하는 구조 설계", "dialogue_logic"),
            ("복합 시스템에서 국소 최적점 함정을 탈출하는 메타-진화 원리 규명", "dialogue_logic"),
        ]
        suite = []
        for idx, (p_text, mode) in enumerate(prompts):
            sp = self.butler.encode_guest_request_to_spike(
                p_text, domain_mode=mode, complexity_tier=3 + (idx % 3)
            )
            suite.append(sp)
        return suite

    # ================================================================== 체크포인트 ③
    def _state_dict(self) -> Dict[str, Any]:
        return {
            "format_version": 2,
            "current_epoch": self.current_epoch,
            "active_dna_id": self.active_dna_id,
            "floor": self.floor,
            "brain": self.brain,
            "press_bank": self.press_bank,
            "ast_coder": self.ast_coder,
            "nightmare": self.nightmare,
            "butler": self.butler,
            "guest_rng": self.guest_rng,
            "history": self.history,
            "recent_briefings": self.recent_briefings,
            "sessions": self.sessions,
        }

    def save_checkpoint(self) -> None:
        p_fn, i_fn = self.brain.dna_plasticity_kernel, self.brain.dna_ironing_kernel
        self.brain.dna_plasticity_kernel, self.brain.dna_ironing_kernel = None, None
        try:
            # 이전 정본을 .bak 으로 밀어두고, 새 정본은 임시 파일에 쓴 뒤 원자적으로 교체
            tmp_path = self.checkpoint_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(self._state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
            if os.path.exists(self.checkpoint_path):
                os.replace(self.checkpoint_path, self.checkpoint_path + ".bak")
            os.replace(tmp_path, self.checkpoint_path)
        finally:
            self.brain.dna_plasticity_kernel, self.brain.dna_ironing_kernel = p_fn, i_fn

    def _candidate_checkpoints(self, extra_search_dirs: List[str]) -> List[str]:
        paths = [self.checkpoint_path, self.checkpoint_path + ".bak"]
        for d in extra_search_dirs:
            for name in (CHECKPOINT_NAME, CHECKPOINT_NAME + ".bak"):
                paths.extend(sorted(glob.glob(os.path.join(d, "**", name), recursive=True)))
        return [p for p in dict.fromkeys(paths) if os.path.exists(p)]

    def try_resume(self, extra_search_dirs: Optional[List[str]] = None) -> Optional[str]:
        best_state, best_path = None, None
        for path in self._candidate_checkpoints(extra_search_dirs or []):
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
                if state.get("format_version") != 2:
                    continue
                if best_state is None or state["current_epoch"] > best_state["current_epoch"]:
                    best_state, best_path = state, path
            except Exception as e:
                print(f"[RESUME] 손상된 체크포인트 건너뜀: {path} ({e})")
        if best_state is None:
            return None

        for key in ("current_epoch", "active_dna_id", "floor", "brain", "press_bank", "ast_coder",
                    "nightmare", "butler", "guest_rng", "history", "recent_briefings", "sessions"):
            setattr(self, key, best_state[key])
        self.solver = SuperpositionSteamIronSolver(self.brain, self.press_bank, self.floor)
        self.nightmare.floor = self.floor
        self.butler.floor = self.floor
        self.ast_coder.relocate(self.output_dir)
        ok, p_fn, i_fn, msg = self.ast_coder.compile_and_sandbox_dna(self.ast_coder.champion.source_code)
        if ok:
            self.brain.dna_plasticity_kernel, self.brain.dna_ironing_kernel = p_fn, i_fn
        self.ast_coder.save_active_dna_to_disk(self.ast_coder.champion)
        self.everest_benchmark_spikes = self._build_4domain_everest_suite()
        return best_path

    # ================================================================== 평가 ②
    def _iron_suite(self, spikes: List[EnergySpikeProblem], peek: bool, install: bool,
                    allow_mitosis: bool = False) -> Dict[str, float]:
        flat_wins, instant, energies, strokes = 0, 0, [], []
        for sp in spikes:
            res = self.solver.iron_spike_to_flat_floor(
                sp,
                max_ironing_strokes=14,
                n_superposition_irons=5,
                current_epoch=self.current_epoch,
                install_press_on_success=install,
                allow_oracle_peek=peek,
                allow_mitosis=allow_mitosis,
            )
            flat_wins += int(res["can_sleep_flat"])
            if res["strokes_used"] == 0 and res["used_hydraulic_press"] is not None:
                instant += 1
            energies.append(res["final_spike_height"])
            strokes.append(res["strokes_used"])
        return {
            "sleep_rate_pct": round(100.0 * flat_wins / len(spikes), 2),
            "mean_energy": float(np.mean(energies)),
            "avg_strokes": round(float(np.mean(strokes)), 2),
            "instant_kills": instant,
        }

    def _asi_quotient(self, mean_energy: float) -> float:
        vitals = self.brain.get_brain_vital_signs()
        return round(
            float(
                100.0
                * math.exp(-15.0 * mean_energy)
                * (1.0 + 0.15 * len(self.press_bank.presses))
                * (1.0 + 0.25 * self.press_bank.max_hierarchy_level)
                * (1.0 + 0.08 * vitals["num_cells_arms"])
            ),
            2,
        )

    def evaluate_on_4domain_everest_suite(self, allow_press_install: bool = False) -> Dict[str, Any]:
        # 원조 계기판은 원본과 같은 조건(분열 허용), 나머지 두 시험은 순수 측정
        on = self._iron_suite(self.everest_benchmark_spikes, peek=True, install=allow_press_install, allow_mitosis=True)
        off = self._iron_suite(self.everest_benchmark_spikes, peek=False, install=False)

        unseen_spikes = []
        for _ in range(self.unseen_guests_per_eval):
            prompt, mode = self.butler.random_walk_in_guest(self.guest_rng)
            tier = 3 + int(self.guest_rng.integers(0, 3))
            unseen_spikes.append(self.butler.encode_guest_request_to_spike(prompt, mode, tier))
        unseen = self._iron_suite(unseen_spikes, peek=False, install=False)

        return {
            # 원조 계기판 (족보 ON, 기존 숫자와 같은 정의)
            "everest_sleep_rate_pct": on["sleep_rate_pct"],
            "mean_residual_spike_energy": round(on["mean_energy"], 5),
            "avg_ironing_strokes": on["avg_strokes"],
            "instant_001s_press_kills": on["instant_kills"],
            "effective_asi_quotient": self._asi_quotient(on["mean_energy"]),
            # 족보 OFF (같은 12문제)
            "oracle_off_sleep_rate_pct": off["sleep_rate_pct"],
            "oracle_off_mean_energy": round(off["mean_energy"], 5),
            "oracle_off_avg_strokes": off["avg_strokes"],
            "oracle_off_instant_kills": off["instant_kills"],
            # 처음 보는 손님 (족보 OFF, 프레스 설치 없음)
            "unseen_guest_sleep_rate_pct": unseen["sleep_rate_pct"],
            "unseen_guest_mean_energy": round(unseen["mean_energy"], 5),
            "unseen_guest_avg_strokes": unseen["avg_strokes"],
            "unseen_asi_quotient": self._asi_quotient(unseen["mean_energy"]),
        }

    # ================================================================== 통역사 근무
    def _serve_walk_in_guests(self) -> List[Dict[str, Any]]:
        served = []
        for _ in range(self.guests_per_epoch):
            prompt, mode = self.butler.random_walk_in_guest(self.guest_rng)
            tier = 3 + int(self.guest_rng.integers(0, 4))
            out = self.butler.serve_guest(
                prompt, self.solver, self.active_dna_id, domain_mode=mode,
                complexity_tier=tier, current_epoch=self.current_epoch, learn_from_guest=True,
            )
            rec = {
                "epoch": self.current_epoch,
                "domain": out["domain_mode"],
                "guest_prompt": out["guest_prompt"],
                "tier": tier,
                "butler_speech": out["butler_speech"],
                "telemetry": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out["iron_telemetry"].items()},
            }
            served.append(rec)
            if self.current_epoch % 10 == 0 or self.current_epoch <= 20:
                with open(self.guest_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.recent_briefings = (self.recent_briefings + served)[-40:]
        return served

    # ================================================================== 진화 루프
    def _stage_label(self, vitals: Dict[str, Any]) -> str:
        if vitals["num_cells_arms"] <= 2:
            return "Amoeba"
        if self.press_bank.max_hierarchy_level <= 1:
            return "Multi-Armed Monster"
        if self.ast_coder.total_novel_genes_committed > 0:
            return "Gene-Inventing Super-Organism"
        return "Self-Coding Super-Organism"

    def _make_record(self, ep: int, stage: str, dna_info: Optional[Dict[str, Any]], eval_ep: Dict[str, Any],
                     elapsed: float) -> Dict[str, Any]:
        vitals = self.brain.get_brain_vital_signs()
        return {
            "epoch": ep,
            "stage_name": stage,
            "num_cells_arms": vitals["num_cells_arms"],
            "total_parameters": vitals["total_parameters"],
            "mitosis_count": vitals["mitosis_count"],
            "apoptosis_count": vitals["apoptosis_count"],
            "ceiling_press_count": len(self.press_bank.presses),
            "presses_installed_total": self.press_bank.total_presses_installed,
            "presses_retired_total": self.press_bank.total_presses_retired,
            "max_press_hierarchy": self.press_bank.max_hierarchy_level,
            "nightmare_tier": self.nightmare.current_nightmare_tier,
            "active_dna_id": self.active_dna_id,
            "ast_node_count": dna_info["ast_node_count"] if dna_info else self.ast_coder.champion.ast_node_count,
            "gene_bank_size": self.ast_coder.gene_bank_size,
            "novel_genes_total": self.ast_coder.total_novel_genes_committed,
            "champion_changed": bool(dna_info and dna_info["champion_changed"]),
            "dna_fitness": dna_info["fitness"] if dna_info else None,
            "guests_served_total": self.butler.guests_served,
            "elapsed_sec": round(elapsed, 2),
            **eval_ep,
        }

    def _print_record(self, rec: Dict[str, Any]) -> None:
        print(
            f"[Epoch {rec['epoch']:05d} | {rec['stage_name'][:29]:29s}] Arms {rec['num_cells_arms']:2d} | "
            f"Presses {rec['ceiling_press_count']:5d} (L{rec['max_press_hierarchy']}) | "
            f"AST {rec['ast_node_count']:4d} | GeneBank {rec['gene_bank_size']:3d} | "
            f"E(on/off/unseen) {rec['mean_residual_spike_energy']:.4f}/{rec['oracle_off_mean_energy']:.4f}/"
            f"{rec['unseen_guest_mean_energy']:.4f} | "
            f"Sleep% {rec['everest_sleep_rate_pct']:5.1f}/{rec['oracle_off_sleep_rate_pct']:5.1f}/"
            f"{rec['unseen_guest_sleep_rate_pct']:5.1f} | Kills {rec['instant_001s_press_kills']:2d}/12 | "
            f"ASI-Q {rec['effective_asi_quotient']:.1f} | {rec['elapsed_sec']:.1f}s",
            flush=True,
        )

    def run_evolution_epochs(
        self,
        num_epochs: int = 6,
        max_hours: Optional[float] = None,
        checkpoint_every_sec: float = 120.0,
        report_every_sec: float = 600.0,
    ) -> Dict[str, Any]:
        run_start = time.time()
        deadline = run_start + max_hours * 3600.0 if max_hours else None
        self.sessions.append({"started_at_epoch": self.current_epoch, "max_hours": max_hours, "num_epochs": num_epochs})

        print("=" * 100)
        print("[PROJECT OUROBOROS v2] AMOEBA -> ASI | MORPHOGENESIS + AST GENE INVENTION + NIGHTMARE SELECTION + BUTLER")
        print(f"  start epoch={self.current_epoch} | budget: epochs={num_epochs}, hours={max_hours}")
        print("=" * 100, flush=True)

        if self.current_epoch == 0 and not self.history:
            eval0 = self.evaluate_on_4domain_everest_suite(allow_press_install=False)
            rec0 = self._make_record(0, "Stage 0: Primordial Single-Cell Amoeba", None, eval0, 0.0)
            self.history.append(rec0)
            self._print_record(rec0)

        last_ckpt = last_report = time.time()
        epoch_times: List[float] = []
        start_ep = self.current_epoch + 1
        for ep in range(start_ep, start_ep + num_epochs):
            if deadline is not None and epoch_times:
                # 다음 에포크 + 마무리 저장까지 들어갈 여유가 없으면 스스로 퇴근
                if time.time() + 1.5 * max(epoch_times[-20:]) + 120.0 > deadline:
                    print(f"[DAEMON] 시간 예산 소진 임박 -> epoch {self.current_epoch}에서 정리합니다.", flush=True)
                    break

            t0 = time.time()
            self.current_epoch = ep

            for _ in range(3):
                self.nightmare.hunt_and_conquer_nightmare(ep, self.solver, self.press_bank)
            if len(self.brain.cells) < min(12, 1 + ep * 2):
                self.brain.trigger_mitosis_neurogenesis(ep, reason=f"epoch_{ep}_everest_expansion")

            # ④ 악몽 × 셀프코더: 사각지대 악몽들이 유전자 선택압이 된다
            eval_spikes = self.everest_benchmark_spikes[:2] + self.nightmare.recent_nightmares(4)
            dna_info = self.ast_coder.evolve_and_hotswap_dna(self.brain, self.solver, eval_spikes, next_gen=ep)
            self.active_dna_id = dna_info["selected_dna_id"]

            self._serve_walk_in_guests()

            eval_ep = self.evaluate_on_4domain_everest_suite(allow_press_install=(ep >= 2))
            self.press_bank.retire_idle_presses(self.max_presses)

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            rec = self._make_record(ep, self._stage_label(self.brain.get_brain_vital_signs()), dna_info, eval_ep, elapsed)
            self.history.append(rec)

            if ep <= start_ep + 19 or ep % 25 == 0 or rec["champion_changed"]:
                self._print_record(rec)
                if dna_info["novel_genes_committed"]:
                    for g in dna_info["novel_genes_committed"]:
                        print(f"    [GENE INVENTED] {g}", flush=True)

            now = time.time()
            if now - last_ckpt >= checkpoint_every_sec:
                self.save_checkpoint()
                last_ckpt = now
            if now - last_report >= report_every_sec:
                self.write_telemetry_and_dashboard()
                last_report = now

        self.save_checkpoint()
        payload = self.write_telemetry_and_dashboard(with_demos=True)
        print(f"[DAEMON] 세션 종료: epoch {self.current_epoch}, 총 {time.time() - run_start:.0f}s", flush=True)
        return payload

    # ================================================================== 결과물
    def _domain_demos(self) -> List[Dict[str, Any]]:
        demos = []
        for sample_spike in [self.everest_benchmark_spikes[i] for i in (0, 3, 6, 9)]:
            iron_res = self.solver.iron_spike_to_flat_floor(
                sample_spike, current_epoch=self.current_epoch,
                install_press_on_success=False, allow_oracle_peek=False, allow_mitosis=False,
            )
            decoded = self.butler.decode_ironed_mold_for_guest(
                sample_spike, iron_res, self.brain.get_brain_vital_signs(), self.active_dna_id
            )
            demos.append({
                "domain": decoded["domain_mode"],
                "butler_speech": decoded["butler_speech"],
                "telemetry": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in decoded["iron_telemetry"].items()},
            })
        return demos

    def write_telemetry_and_dashboard(self, with_demos: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "final_epoch": self.current_epoch,
            "final_brain_vitals": self.brain.get_brain_vital_signs(),
            "final_dna_id": self.active_dna_id,
            "final_dna_file": self.ast_coder.dna_file_path,
            "gene_bank": {
                "plasticity": self.ast_coder.plasticity_genes,
                "ironing": self.ast_coder.ironing_genes,
                "births": self.ast_coder.gene_births,
            },
            "sessions": self.sessions,
            "history": self.history,
            "recent_butler_briefings": self.recent_briefings[-12:],
        }
        if with_demos:
            payload["domain_demos"] = self._domain_demos()
        with open(self.telemetry_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=float)
        self._render_evolution_dashboard()
        payload["chart_path"] = self.chart_path
        payload["telemetry_path"] = self.telemetry_path
        return payload

    def _render_evolution_dashboard(self) -> None:
        H = self.history
        epochs = [h["epoch"] for h in H]
        dense = len(epochs) > 60
        mk = dict(marker=None) if dense else dict(marker="o", markersize=5)

        fig, axes = plt.subplots(2, 3, figsize=(19.5, 9.6), dpi=130)
        fig.patch.set_facecolor("#090d16")
        for ax in axes.flat:
            ax.set_facecolor("#131c2e")
            ax.tick_params(colors="#e2e8f0")
            for spine in ax.spines.values():
                spine.set_color("#334155")
            ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
            ax.set_xlabel("Evolution Epoch", color="#cbd5e1")

        def leg(ax):
            ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8)

        def title(ax, t):
            ax.set_title(t, color="#f8fafc", fontsize=10.5, fontweight="bold")

        ax = axes[0, 0]
        ax.plot(epochs, [h["effective_asi_quotient"] for h in H], color="#38bdf8", linewidth=2.4, label="ASI-Q (benchmark, oracle ON)", **mk)
        ax.plot(epochs, [h["unseen_asi_quotient"] for h in H], color="#f472b6", linewidth=2.0, label="ASI-Q (unseen guests, oracle OFF)", **mk)
        ax.set_yscale("log")
        title(ax, "Exponential ASI Quotient (log scale)")
        leg(ax)

        ax = axes[0, 1]
        ax.plot(epochs, [h["num_cells_arms"] for h in H], color="#a855f7", linewidth=2.2, label="Morphogenetic Arms (Cells)", **mk)
        ax2 = ax.twinx()
        ax2.plot(epochs, [h["ceiling_press_count"] for h in H], color="#4ade80", linewidth=2.2, label="Ceiling Presses (right)")
        ax2.tick_params(colors="#4ade80")
        title(ax, "Morphogenetic Growth: Arms & Hydraulic Presses")
        leg(ax)
        ax2.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8)

        ax = axes[0, 2]
        ax.plot(epochs, [h["ast_node_count"] for h in H], color="#fb923c", linewidth=2.2, label="Champion AST Nodes", **mk)
        ax3 = ax.twinx()
        ax3.plot(epochs, [h["gene_bank_size"] for h in H], color="#facc15", linewidth=2.0, linestyle="--", label="Gene Bank Size (right)")
        ax3.tick_params(colors="#facc15")
        title(ax, "Gene Buffet Refill: Self-Invented AST Genes")
        leg(ax)
        ax3.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8)

        ax = axes[1, 0]
        ax.plot(epochs, [h["avg_ironing_strokes"] for h in H], color="#fbbf24", linewidth=2.4, label="Oracle ON", **mk)
        ax.plot(epochs, [h["oracle_off_avg_strokes"] for h in H], color="#f87171", linewidth=2.0, label="Oracle OFF", **mk)
        ax.plot(epochs, [h["unseen_guest_avg_strokes"] for h in H], color="#f472b6", linewidth=1.6, label="Unseen guests", **mk)
        ax.set_ylabel("Strokes (0 = Instant Press Kill!)", color="#cbd5e1")
        title(ax, "Laziness Triumph: Avg Steam-Iron Strokes per Everest")
        leg(ax)

        ax = axes[1, 1]
        ax.plot(epochs, [h["mean_residual_spike_energy"] for h in H], color="#38bdf8", linewidth=2.4, label="Oracle ON", **mk)
        ax.plot(epochs, [h["oracle_off_mean_energy"] for h in H], color="#f43f5e", linewidth=2.0, label="Oracle OFF", **mk)
        ax.plot(epochs, [h["unseen_guest_mean_energy"] for h in H], color="#f472b6", linewidth=1.6, label="Unseen guests", **mk)
        ax.axhline(y=self.floor.flat_tolerance, color="#22c55e", linestyle=":", linewidth=1.8, label="Flat Sleep Threshold")
        ax.set_ylabel("Contradiction Spike Energy", color="#cbd5e1")
        title(ax, "Residual Everest Spike Height (E -> 0 Flat Floor)")
        leg(ax)

        ax = axes[1, 2]
        ax.plot(epochs, [h["everest_sleep_rate_pct"] for h in H], color="#38bdf8", linewidth=2.4, label="Benchmark, Oracle ON", **mk)
        ax.plot(epochs, [h["oracle_off_sleep_rate_pct"] for h in H], color="#f43f5e", linewidth=2.0, label="Benchmark, Oracle OFF", **mk)
        ax.plot(epochs, [h["unseen_guest_sleep_rate_pct"] for h in H], color="#f472b6", linewidth=1.6, label="Unseen guests", **mk)
        ax.set_ylim(-3, 103)
        ax.set_ylabel("Flat-Sleep Rate (%)", color="#cbd5e1")
        title(ax, "Can the Monster Sleep? (Oracle ON vs OFF vs Unseen)")
        leg(ax)

        fig.suptitle(
            f"PROJECT OUROBOROS v2: Amoeba (1 Cell) -> Gene-Inventing Self-Coding Engine | epoch {self.current_epoch}",
            color="#ffffff", fontsize=13.5, fontweight="bold", y=0.985,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.chart_path, facecolor=fig.get_facecolor())
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="PROJECT OUROBOROS infinite evolution daemon")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hours", type=float, default=None, help="시간 예산 (Kaggle은 11.0 권장)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--resume-from", action="append", default=[], help="이전 체크포인트를 찾을 폴더 (여러 번 지정 가능)")
    args = ap.parse_args()

    daemon = OuroborosInfiniteEvolutionDaemon(output_dir=args.out, seed=args.seed)
    resumed = daemon.try_resume(args.resume_from)
    if resumed:
        print(f"[RESUME] 이어받기 성공: {resumed} (epoch {daemon.current_epoch}, 팔 {len(daemon.brain.cells)}개, "
              f"프레스 {len(daemon.press_bank.presses)}대, 유전자 은행 {daemon.ast_coder.gene_bank_size}개)")

    res = daemon.run_evolution_epochs(num_epochs=args.epochs, max_hours=args.hours)
    print("\n" + "=" * 100)
    print("[4-DOMAIN EVEREST LIVE DEMONSTRATION AFTER EVOLUTION (oracle OFF)]")
    print("=" * 100)
    for demo in res["domain_demos"]:
        print(demo["butler_speech"])
        print("-" * 100)


if __name__ == "__main__":
    main()
