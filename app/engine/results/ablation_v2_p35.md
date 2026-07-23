| arm | params | ce loss ↓ | next-token acc ↑ | drift horizon ↑ (PSNR>35dB, /24) |
| --- | ---: | ---: | ---: | ---: |
| baseline (ssm) | 0.64M | 1.605 | 60.2% | 0 |
| + mamba block | 0.84M | 0.842 | 79.8% | 0 |
| + film actions | 0.84M | 0.844 | 79.8% | 0 |
| + noise aug (full) | 0.84M | 1.030 | 74.4% | 0 |

Full stack vs baseline: drift horizon 0 → 0 frames, next-token accuracy 60.2% → 74.4%.
