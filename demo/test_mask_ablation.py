"""
Test mask ablation across 5 cases:
1. Assert mask T-only differs from mask DTR.
2. Assert disabled shares are 0 (0%).
3. Assert active shares sum to 100% (1.0).
4. Assert cached == live within 1e-4.
"""
import os
import sys
import json
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from demo.export.load_models import load
from demo.export.explain_core import (
    build_case_input_from_json,
    _run_full_diagnosis_core,
    _get_vectorizer,
)

CASES_TO_TEST = ["CASE-0002", "CASE-0069", "CASE-0151", "CASE-0201", "CASE-0262"]
MODELS = ["conflux", "quorum", "nexus"]

def main():
    print("Loading models and vectorizer...")
    loaded_models = {mid: load(mid) for mid in MODELS}
    vectorizer = _get_vectorizer()

    passed_count = 0

    for cid in CASES_TO_TEST:
        case_path = os.path.join(ROOT_DIR, "demo", "web", "public", "data", "cases", f"{cid}.json")
        assert os.path.exists(case_path), f"Case file not found: {case_path}"
        with open(case_path, "r", encoding="utf-8") as f:
            case_data = json.load(f)

        case_input = build_case_input_from_json(case_data)
        active_text = case_input.pop("_active_text", "")
        true_cls = case_data["true_class"]
        elem = case_data["element_name"]

        print(f"\nTesting {cid} ({true_cls}):")

        for mid in MODELS:
            m = loaded_models[mid]

            # 1. Live DTR
            live_dtr = _run_full_diagnosis_core(
                model_id=mid,
                predictor=m,
                case_input=case_input,
                true_class=true_cls,
                element_name=elem,
                report_text=active_text,
                vectorizer=vectorizer,
                mask=("D", "T", "R"),
            )

            # 2. Live T-only
            live_t = _run_full_diagnosis_core(
                model_id=mid,
                predictor=m,
                case_input=case_input,
                true_class=true_cls,
                element_name=elem,
                report_text=active_text,
                vectorizer=vectorizer,
                mask=("T",),
            )

            # Assertion 1: T-only differs from DTR in probabilities
            dtr_probs = np.array(list(live_dtr["prediction"]["probs"].values()))
            t_probs = np.array(list(live_t["prediction"]["probs"].values()))
            prob_diff = float(np.max(np.abs(dtr_probs - t_probs)))
            assert prob_diff > 1e-4, f"[{cid}/{mid}] Expected T-only probs to differ from DTR, but max diff={prob_diff:.6f}"

            # Assertion 2: Disabled shares are 0 in T-only
            contrib_t = live_t["explanation"]["contrib"]
            assert contrib_t["D"]["share"] == 0.0, f"[{cid}/{mid}] Expected D share=0.0 in T-only, got {contrib_t['D']['share']}"
            assert contrib_t["R"]["share"] == 0.0, f"[{cid}/{mid}] Expected R share=0.0 in T-only, got {contrib_t['R']['share']}"

            # Assertion 3: Active shares sum to 100% (1.0)
            active_sum = contrib_t["T"]["share"]
            assert abs(active_sum - 1.0) < 1e-4, f"[{cid}/{mid}] Expected active share sum=1.0, got {active_sum}"

            # Also check DT pair: disabled R is 0, active D+T sum to 1.0
            live_dt = _run_full_diagnosis_core(
                model_id=mid,
                predictor=m,
                case_input=case_input,
                true_class=true_cls,
                element_name=elem,
                report_text=active_text,
                vectorizer=vectorizer,
                mask=("D", "T"),
            )
            contrib_dt = live_dt["explanation"]["contrib"]
            assert contrib_dt["R"]["share"] == 0.0, f"[{cid}/{mid}] Expected R share=0.0 in DT, got {contrib_dt['R']['share']}"
            dt_sum = contrib_dt["D"]["share"] + contrib_dt["T"]["share"]
            assert abs(dt_sum - 1.0) < 1e-4, f"[{cid}/{mid}] Expected DT active shares sum=1.0, got {dt_sum}"

            # Assertion 4: Cached == Live within 1e-4
            if "by_mask" in case_data:
                cached_t = case_data["by_mask"][mid]["T"]
                cached_dtr = case_data["by_mask"][mid]["DTR"]

                # Check confidence match
                conf_diff_t = abs(cached_t["prediction"]["confidence"] - live_t["prediction"]["confidence"])
                assert conf_diff_t < 1e-4, f"[{cid}/{mid}] Cached vs Live T conf diff={conf_diff_t:.6f}"

                conf_diff_dtr = abs(cached_dtr["prediction"]["confidence"] - live_dtr["prediction"]["confidence"])
                assert conf_diff_dtr < 1e-4, f"[{cid}/{mid}] Cached vs Live DTR conf diff={conf_diff_dtr:.6f}"

                # Check Shapley match
                sh_diff_t = abs(cached_t["explanation"]["contrib"]["T"]["share"] - contrib_t["T"]["share"])
                assert sh_diff_t < 1e-4, f"[{cid}/{mid}] Cached vs Live T share diff={sh_diff_t:.6f}"

            print(f"  [PASS] {mid.upper():7s}: prob_diff={prob_diff:.3f}, DTR_pred={live_dtr['prediction']['pred']}, T_pred={live_t['prediction']['pred']}, T_share={contrib_t['T']['share']*100:.0f}%")
            passed_count += 1

    print(f"\n==========================================")
    print(f"ALL {passed_count} MASK ABLATION TESTS PASSED (5 cases x 3 models)")
    print(f"==========================================")

if __name__ == "__main__":
    main()
