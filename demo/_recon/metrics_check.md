# Metrics Reproduction Check

Verification per Phase 1 Section 2.5: Exported models reproduce reported `final_results` test metrics on the full 158-case test set within ±0.5 percentage points.

| Model | Fusion Architecture | Reported Accuracy | Exported Accuracy | Delta Acc | Reported Macro-F1 | Exported Macro-F1 | Delta F1 | Gate Status |
|---|---|---|---|---|---|---|---|---|
| **CONFLUX** | Early Fusion (Input Concat 384-d) | 87.97% (0.8797) | 87.97% (0.8797) | +0.00% | 82.20% (0.8220) | 82.20% (0.8220) | +0.00% | **PASSED ✓** |
| **QUORUM** | Late Fusion (Weighted Vote) | 83.54% (0.8354) | 83.54% (0.8354) | +0.00% | 76.51% (0.7651) | 76.51% (0.7651) | +0.00% | **PASSED ✓** |
| **NEXUS** | Hybrid Fusion (Cross-Modal + Tel) | 82.28% (0.8228) | 82.28% (0.8228) | +0.00% | 73.03% (0.7303) | 73.03% (0.7303) | +0.00% | **PASSED ✓** |

## Curated Demo Set vs Full Test Set Accuracy

Curated test cases: 40 cases (seed 13, stratified across all 4 classes).

| Model | Full Test Set (158 cases) | Curated Demo Set (40 cases) |
|---|---|---|
| **CONFLUX** | 87.97% | 90.00% |
| **QUORUM** | 83.54% | 85.00% |
| **NEXUS** | 82.28% | 82.50% |

All metrics match within ±0.5 percentage points. **HARD GATE A & B PASSED.**
