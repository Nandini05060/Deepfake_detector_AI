# Phase 12 Candidate Training Log Summary

**Candidate Model:** `LXDFD-MultiScale-12`
**Checkpoint Path:** `experiments\phase12_multiscale_v1\lxfd_multiscale12_best.pt`
**Checkpoint SHA-256:** `a2b9989205f255248ca7c378e25164b88fbc47691ce8b9b9b815fcd00a72e773`
**Best Validation Epoch:** 3
**Best Validation ROC-AUC:** 0.9511
**Total Training Duration:** 369.2s

| Epoch | Train Loss | Train AUC | Val Loss | Val AUC | Val Accuracy | Val F1 | Duration |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.1831 | 0.9793 | 0.3270 | 0.9438 | 87.2% | 0.8775 | 34.8s |
| 2 | 0.1138 | 0.9934 | 0.3381 | 0.9437 | 88.4% | 0.8889 | 29.4s |
| 3 (*) | 0.0942 | 0.9948 | 0.3597 | 0.9511 | 86.8% | 0.8788 | 37.5s |
| 4 | 0.0822 | 0.9952 | 0.5016 | 0.9243 | 82.2% | 0.8413 | 47.5s |
| 5 | 0.0719 | 0.9954 | 0.4083 | 0.9416 | 86.0% | 0.8702 | 49.0s |
| 6 | 0.0422 | 0.9996 | 0.4256 | 0.9452 | 87.2% | 0.8830 | 58.3s |
| 7 | 0.0492 | 0.9988 | 0.4172 | 0.9383 | 85.1% | 0.8636 | 57.9s |
| 8 | 0.0488 | 0.9988 | 0.3735 | 0.9435 | 87.6% | 0.8828 | 54.7s |
