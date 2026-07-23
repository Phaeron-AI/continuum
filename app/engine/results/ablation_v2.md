| arm | params | ce loss ↓ | next-token acc ↑ | drift horizon ↑ (PSNR>20dB, /24) |
| --- | ---: | ---: | ---: | ---: |
| baseline (ssm) | 0.64M | 1.612 | 59.9% | 24 |
| + mamba block | 0.84M | 0.842 | 79.8% | 24 |
| + film actions | 0.84M | 0.845 | 79.8% | 24 |
| + noise aug (full) | 0.84M | 1.030 | 74.4% | 24 |

Full stack vs baseline: drift horizon 24 → 24 frames, next-token accuracy 59.9% → 74.4%.
