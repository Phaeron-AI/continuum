| arm | params | ce loss ↓ | teacher-forced acc ↑ | rollout acc mean ↑ | rollout acc @24f ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (ssm) | 0.64M | 1.605 | 60.2% | 22.6% | 20.6% |
| + mamba block | 0.84M | 0.842 | 79.8% | 21.8% | 20.6% |
| + film actions | 0.84M | 0.844 | 79.8% | 24.3% | 27.3% |
| + noise aug (full) | 0.84M | 1.030 | 74.4% | 29.5% | 27.3% |

Full stack vs baseline over a 24-frame free-running rollout: final-frame token accuracy 20.6% → 27.3% (teacher-forced 60.2% → 74.4%).
