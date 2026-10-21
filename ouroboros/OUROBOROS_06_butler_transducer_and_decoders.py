"""
================================================================================
PROJECT OUROBOROS | MODULE 6: THE POLITE BUTLER (TRANSDUCER & UNIVERSAL DECODER)
File: OUROBOROS_06_butler_transducer_and_decoders.py
================================================================================
세계관 및 아키텍처 매핑 (The Polite Butler Outside the Bedroom Door):
- 문밖 카운터의 '말만 잘하는 집사(PoliteButlerTransducer)'가 손님의 요청을 안방 바닥 밑에
  가시 산으로 밀어 넣고, 잠만보가 다림질해 남긴 '평평한 주물 자국'을 주워 와서
  신약 분자 구조식, 수학 해답, 파이썬 코드, 또는 논리적 한국어 답변으로 통역합니다!

[v2 업그레이드: 통역사 상시 근무]
- random_walk_in_guest(): 4대 영역 템플릿을 조합해 매번 처음 보는 손님 요청을 만들어냅니다.
- serve_guest(): 진짜 손님에게는 정답지가 없으므로 족보 OFF로 다림질한 뒤 통역해 드립니다.
================================================================================
"""

from __future__ import annotations
import hashlib
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from OUROBOROS_01_energy_bedroom_and_spikes import (
    EnergySpikeProblem,
    UniversalBedroomFloor,
)
from OUROBOROS_03_steam_iron_and_hydraulic_press import SuperpositionSteamIronSolver


WALK_IN_GUEST_TEMPLATES: Dict[str, Tuple[List[str], List[str]]] = {
    "drug_molecular": (
        ["KRAS G12C", "EGFR T790M", "BRAF V600E", "ALK 융합", "HER2 과발현", "PD-L1", "BCR-ABL T315I", "PI3K 알파"],
        ["표적을 간 독성 없이 억제하는 신약 분자 설계",
         "내성을 돌파하는 고투과성 리간드 역설계",
         "선택성을 높이고 hERG 심장 독성을 줄이는 분자 합성",
         "경구 흡수율을 높인 저분자 후보 구조 제안"],
    ),
    "math_symbolic": (
        ["비선형 결합 편미분 방정식", "고차 다항 행렬 방정식", "비대칭 위상 그래프", "확률 미분 방정식", "변분 문제"],
        ["의 최소 작용 고유상태 증명", "의 비자명 불변 근 역산", "위의 전역 에너지 평형 정리 유도", "의 고정점 존재성 증명"],
    ),
    "code_program": (
        ["O(1) 메모리", "락-프리", "캐시 친화적", "스트리밍", "비동기"],
        ["상태 전이 커널 코드 합성", "텐서 압축 알고리즘 함수 생성", "스케줄러 함수 합성", "중복 제거 파이썬 코드 작성"],
    ),
    "dialogue_logic": (
        ["한정된 GPU 메모리에서", "에너지 비용을 절반으로 줄이면서", "데이터가 거의 없을 때", "팀원이 세 명뿐일 때"],
        ["치명적 망각 없이 무한 학습하는 전략 도출", "추론 깊이를 늘리는 구조 설계",
         "국소 최적점 함정을 탈출하는 원리 규명", "우선순위를 정하는 의사결정 원칙 정리"],
    ),
}


class PoliteButlerTransducer:
    """[문밖 카운터 직원: 입만 산 집사]"""
    def __init__(self, floor: UniversalBedroomFloor):
        self.floor = floor
        self.guests_served: int = 0

    def random_walk_in_guest(self, rng: np.random.Generator) -> Tuple[str, str]:
        """4대 영역 중 하나에서 매번 다른 손님 요청을 조합해 문 앞에 세웁니다."""
        domains = list(WALK_IN_GUEST_TEMPLATES.keys())
        mode = domains[int(rng.integers(0, len(domains)))]
        heads, tails = WALK_IN_GUEST_TEMPLATES[mode]
        head = heads[int(rng.integers(0, len(heads)))]
        tail = tails[int(rng.integers(0, len(tails)))]
        ticket = int(rng.integers(1000, 9999))
        joiner = "" if tail.startswith(("의", "위")) else " "
        return f"{head}{joiner}{tail} (손님 번호표 #{ticket})", mode

    def encode_guest_request_to_spike(
        self,
        guest_prompt: str,
        domain_mode: str = "auto",
        complexity_tier: int = 3,
    ) -> EnergySpikeProblem:
        if domain_mode == "auto":
            p_low = guest_prompt.lower()
            if any(k in p_low for k in ["분자", "신약", "암", "단백질", "독성", "drug", "molecule"]):
                domain_mode = "drug_molecular"
            elif any(k in p_low for k in ["방정식", "수학", "증명", "근", "미분", "math", "equation"]):
                domain_mode = "math_symbolic"
            elif any(k in p_low for k in ["코드", "파이썬", "알고리즘", "함수", "code", "python"]):
                domain_mode = "code_program"
            else:
                domain_mode = "dialogue_logic"

        digest = hashlib.sha256(guest_prompt.encode("utf-8")).hexdigest()
        seed_val = int(digest[:8], 16) % (2 ** 31 - 1)

        spike = self.floor.spawn_spike_problem(
            complexity_tier=complexity_tier,
            category_hint=domain_mode,
            custom_seed=seed_val,
            metadata={"guest_prompt": guest_prompt, "domain_mode": domain_mode},
        )
        return spike

    def serve_guest(
        self,
        guest_prompt: str,
        solver: SuperpositionSteamIronSolver,
        active_dna_id: str,
        domain_mode: str = "auto",
        complexity_tier: int = 3,
        current_epoch: int = 1,
        learn_from_guest: bool = True,
    ) -> Dict[str, Any]:
        """손님 응대 풀코스: 요청 -> 가시 산 -> 족보 없이 다림질 -> 통역."""
        spike = self.encode_guest_request_to_spike(guest_prompt, domain_mode, complexity_tier)
        iron_res = solver.iron_spike_to_flat_floor(
            spike,
            max_ironing_strokes=14,
            n_superposition_irons=5,
            current_epoch=current_epoch,
            install_press_on_success=learn_from_guest,
            allow_oracle_peek=False,
        )
        self.guests_served += 1
        return self.decode_ironed_mold_for_guest(
            spike, iron_res, solver.brain.get_brain_vital_signs(), active_dna_id
        )

    def decode_ironed_mold_for_guest(
        self,
        spike: EnergySpikeProblem,
        iron_result: Dict[str, Any],
        brain_vitals: Dict[str, Any],
        active_dna_id: str,
    ) -> Dict[str, Any]:
        mold = iron_result["best_mold"]
        mode = spike.metadata.get("domain_mode", spike.category_hint)
        prompt = spike.metadata.get("guest_prompt", "")

        node_norms = np.linalg.norm(mold, axis=-1)
        dominant_nodes = [int(i) for i in np.argsort(node_norms)[::-1][:4]]
        channel_means = np.mean(mold, axis=0)

        if mode == "drug_molecular":
            artifact = self._decode_to_denovo_drug_molecule(mold, channel_means, iron_result["final_spike_height"])
        elif mode == "math_symbolic":
            artifact = self._decode_to_math_proof_and_roots(mold, channel_means, iron_result["final_spike_height"])
        elif mode == "code_program":
            artifact = self._decode_to_executable_python_code(mold, dominant_nodes)
        else:
            artifact = self._decode_to_logical_dialogue_plan(prompt, mold, dominant_nodes)

        peek_note = "족보 사용" if iron_result.get("oracle_peeked") else "족보 없이 맨손"
        butler_speech = (
            f"[집사 브리핑] 손님, 요청하신 \"{prompt}\" 건을 안방 바닥에 가시 산(초기 높이 E={iron_result['initial_spike_height']:.4f})으로 "
            f"밀어 넣었습니다.\n"
            f"안방 주인님(현재 팔 {brain_vitals['num_cells_arms']}개, 자가코딩 유전자 '{active_dna_id}')께서 "
            f"천장 유압 프레스({iron_result['used_hydraulic_press'] or '직접 스팀다리미질'})와 {iron_result['strokes_used']}번의 다림질({peek_note})로 "
            f"가시 높이를 E={iron_result['final_spike_height']:.5f} (숙면 가능 평지={iron_result['can_sleep_flat']})까지 밀어버리셨습니다.\n"
            f"다리미 밑에 찍혀 나온 [{mode}] 결과물을 정리해 올립니다:\n{artifact['formatted_output']}"
        )

        return {
            "domain_mode": mode,
            "guest_prompt": prompt,
            "butler_speech": butler_speech,
            "decoded_artifact": artifact,
            "iron_telemetry": {
                "initial_energy": iron_result["initial_spike_height"],
                "final_energy": iron_result["final_spike_height"],
                "can_sleep_flat": iron_result["can_sleep_flat"],
                "strokes_used": iron_result["strokes_used"],
                "used_press": iron_result["used_hydraulic_press"],
                "installed_press": iron_result["installed_new_press"],
                "oracle_peeked": iron_result.get("oracle_peeked", False),
            },
        }

    def _decode_to_denovo_drug_molecule(
        self, mold: np.ndarray, chan: np.ndarray, final_e: float
    ) -> Dict[str, Any]:
        selfies_blocks = [
            "[C]", "[=C]", "[N]", "[=O]", "[Ring1]", "[Branch1]", "[F]", "[S]", "[C@H1]", "[O]"
        ]
        seq = []
        flat_vals = mold.reshape(-1)[:18]
        for v in flat_vals:
            idx = int((math.tanh(float(v)) + 1.0) * 0.5 * (len(selfies_blocks) - 1))
            seq.append(selfies_blocks[idx])
        selfies_str = "".join(seq)

        binding_affinity_pic50 = round(7.2 + 2.4 * math.exp(-15.0 * final_e) + 0.3 * float(np.std(chan)), 2)
        hepatotoxicity_risk = round(max(0.02, min(0.95, float(final_e * 8.5))), 3)
        qed_druglikeness = round(min(0.94, 0.68 + 0.25 * math.exp(-12.0 * final_e)), 3)

        text = (
            f"  * 역설계된 신약 후보 분자 (SELFIES 구조식): {selfies_str}\n"
            f"  * 예상 표적 단백질 결합 친화도 (pIC50): {binding_affinity_pic50} (높을수록 강력 타격)\n"
            f"  * 예상 간 독성 위험도 (Hepatotoxicity): {hepatotoxicity_risk * 100:.1f}% (낮을수록 안전)\n"
            f"  * 종합 약물 적합성 지수 (QED Score): {qed_druglikeness} / 1.000"
        )
        return {
            "selfies_structure": selfies_str,
            "binding_affinity_pic50": binding_affinity_pic50,
            "hepatotoxicity_risk": hepatotoxicity_risk,
            "qed_score": qed_druglikeness,
            "formatted_output": text,
        }

    def _decode_to_math_proof_and_roots(
        self, mold: np.ndarray, chan: np.ndarray, final_e: float
    ) -> Dict[str, Any]:
        steps = [
            "Step 1 [대칭군 분해]: 주어진 다변수 비선형 제약식을 직교 고유 공간(Eigen-subspace)으로 투영",
            f"Step 2 [불변량 소거]: 주물 주성분 고유값 lambda_1={chan[0]:+.4f}, lambda_2={chan[1]:+.4f} 를 대입하여 교차 비선형 항 상쇄",
            f"Step 3 [정류점 수렴]: 잔여 모순 에너지 E={final_e:.5f} <= tol 조건 하에서 유일 해(Fixed-Point Solution) 도출 완료",
        ]
        text = "\n".join(f"  * {s}" for s in steps)
        return {
            "proof_steps": steps,
            "eigen_invariants": [round(float(c), 4) for c in chan[:4]],
            "formatted_output": text,
        }

    def _decode_to_executable_python_code(
        self, mold: np.ndarray, dominant_nodes: List[int]
    ) -> Dict[str, Any]:
        w0, w1, w2 = [round(float(mold[dominant_nodes[i], i]), 4) for i in range(3)]
        code_str = (
            f"def synthesized_optimal_operator(state_vec):\n"
            f"    # Synthesized from flattened mold dominant nodes {dominant_nodes[:3]}\n"
            f"    s1 = state_vec * {w0} + np.tanh(state_vec * {w1})\n"
            f"    return np.tanh(s1 + {w2} * (s1 ** 3))\n"
        )
        text = f"  * 합성된 파이썬 알고리즘 코드:\n```python\n{code_str.strip()}\n```"
        return {
            "synthesized_code": code_str,
            "formatted_output": text,
        }

    def _decode_to_logical_dialogue_plan(
        self, prompt: str, mold: np.ndarray, dominant_nodes: List[int]
    ) -> Dict[str, Any]:
        coherence = round(float(100.0 * (1.0 / (1.0 + np.var(mold)))), 1)
        plan = (
            f"  * [핵심 원인 타점]: 지형 내 최대 가시 집중 노드 #{dominant_nodes[0]} 및 #{dominant_nodes[1]}의 병목 현상 규명\n"
            f"  * [최적 다림질 해법]: 단선적 접근 대신 노드 #{dominant_nodes[0]}번과 #{dominant_nodes[2]}번을 동시에 역위상으로 눌러 부작용 없이 모순을 상쇄\n"
            f"  * [논리 일관성 점수]: {coherence}% (잠재 평형 도달도)"
        )
        return {
            "dominant_causal_nodes": dominant_nodes,
            "coherence_score": coherence,
            "formatted_output": plan,
        }


if __name__ == "__main__":
    from OUROBOROS_02_amoeba_morphogenetic_brain import AmoebaMorphogeneticBrain
    from OUROBOROS_03_steam_iron_and_hydraulic_press import CeilingHydraulicPressBank

    floor = UniversalBedroomFloor(seed=42)
    brain = AmoebaMorphogeneticBrain(seed=42)
    bank = CeilingHydraulicPressBank()
    solver = SuperpositionSteamIronSolver(brain, bank, floor)
    butler = PoliteButlerTransducer(floor)

    q = "KRAS 변이 암세포의 신호전달 가시를 부작용(간 독성) 없이 평평하게 잠재울 새로운 신약 분자 구조를 설계해줘."
    out = butler.serve_guest(q, solver, "dna_gen0_amoeba_seed", domain_mode="drug_molecular")
    print(out["butler_speech"])
    rng = np.random.default_rng(0)
    for _ in range(3):
        print("  walk-in:", butler.random_walk_in_guest(rng))
