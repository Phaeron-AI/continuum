| arm | params | ce loss ↓ | teacher-forced acc ↑ | rollout acc mean ↑ | rollout acc @24f ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline (ssm) | 0.64M | 1.605 | 60.2% | 19.8% | 18.4% |
| + mamba block | 0.84M | 0.842 | 79.8% | 16.3% | 16.8% |
| + film actions | 0.84M | 0.844 | 79.8% | 20.4% | 21.5% |
| + noise aug (full) | 0.84M | 1.030 | 74.4% | 27.3% | 27.0% |

Full stack vs baseline over a 24-frame free-running rollout: final-frame token accuracy 18.4% → 27.0% (teacher-forced 60.2% → 74.4%).
