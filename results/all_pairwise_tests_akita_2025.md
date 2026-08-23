# Every pairwise comparison — Akita, 2025

All 55 unordered pairs among the eleven benchmarked methods, at K = 20, on the 323 days with at least one reported sighting.

Δ is the mean per-day difference in Recall@20 (first method minus second); the interval is a day-level paired bootstrap (B = 5,000) and *p* a day-level sign-flip permutation test (P = 5,000), both seeded with 42 — the same procedure as Table 2 of the paper.

Two Bonferroni thresholds are reported, because the answer depends on which family the comparison is read as part of. **This family** is the 55 pairs in this table, α = 0.00091. **Table 2** of the paper corrects over its own family of thirteen, α = 0.0038, and that is the column to use when checking a claim made in the paper. Rows are sorted by *p*.

The two disagree on 4 of the 55 pairs — those with *p* between 0.00091 and 0.0038, which the stricter family-wide threshold rejects. Among them on Akita is HierBayes against GLM-Logit (*p* = 0.0026), the comparison Section 6 describes as significantly worse; it is significant at the paper's threshold and not at this table's.

| Comparison | Δ | 95% CI | *p* | Sig. (this family, α = 0.00091) | Sig. (paper, α = 0.0038) |
|------------|--:|:------:|----:|:------------------:|:-----------------------:|
| B0: Random vs B1: Static prior | -0.3243 | [-0.3538, -0.2939] | 0.0002 | yes | yes |
| B0: Random vs B2: Recent MA | -0.3371 | [-0.3665, -0.3064] | 0.0002 | yes | yes |
| B0: Random vs B3: DoY season | -0.2717 | [-0.2995, -0.2436] | 0.0002 | yes | yes |
| B0: Random vs B4: B1+B3 | -0.3379 | [-0.3669, -0.3082] | 0.0002 | yes | yes |
| B0: Random vs B5: B2+B3 | -0.3468 | [-0.3762, -0.3174] | 0.0002 | yes | yes |
| B0: Random vs ET | -0.2453 | [-0.2735, -0.2177] | 0.0002 | yes | yes |
| B0: Random vs GLM-Logit | -0.3739 | [-0.4026, -0.3454] | 0.0002 | yes | yes |
| B0: Random vs HierBayes | -0.3512 | [-0.3820, -0.3212] | 0.0002 | yes | yes |
| B0: Random vs Poisson-GLM | +0.0771 | [+0.0629, +0.0933] | 0.0002 | yes | yes |
| B0: Random vs TTM | -0.3145 | [-0.3442, -0.2842] | 0.0002 | yes | yes |
| B1: Static prior vs ET | +0.0789 | [+0.0535, +0.1044] | 0.0002 | yes | yes |
| B1: Static prior vs GLM-Logit | -0.0496 | [-0.0718, -0.0281] | 0.0002 | yes | yes |
| B3: DoY season vs HierBayes | -0.0795 | [-0.1092, -0.0516] | 0.0002 | yes | yes |
| B3: DoY season vs B4: B1+B3 | -0.0662 | [-0.0914, -0.0428] | 0.0002 | yes | yes |
| B2: Recent MA vs ET | +0.0917 | [+0.0574, +0.1241] | 0.0002 | yes | yes |
| B1: Static prior vs Poisson-GLM | +0.4014 | [+0.3765, +0.4273] | 0.0002 | yes | yes |
| B3: DoY season vs B5: B2+B3 | -0.0751 | [-0.1020, -0.0479] | 0.0002 | yes | yes |
| B3: DoY season vs GLM-Logit | -0.1022 | [-0.1343, -0.0719] | 0.0002 | yes | yes |
| B2: Recent MA vs Poisson-GLM | +0.4142 | [+0.3895, +0.4394] | 0.0002 | yes | yes |
| GLM-Logit vs Poisson-GLM | +0.4510 | [+0.4259, +0.4765] | 0.0002 | yes | yes |
| Poisson-GLM vs TTM | -0.3917 | [-0.4171, -0.3660] | 0.0002 | yes | yes |
| HierBayes vs TTM | +0.0367 | [+0.0172, +0.0564] | 0.0002 | yes | yes |
| GLM-Logit vs TTM | +0.0594 | [+0.0376, +0.0820] | 0.0002 | yes | yes |
| HierBayes vs Poisson-GLM | +0.4283 | [+0.4037, +0.4546] | 0.0002 | yes | yes |
| B5: B2+B3 vs ET | +0.1014 | [+0.0682, +0.1330] | 0.0002 | yes | yes |
| B5: B2+B3 vs Poisson-GLM | +0.4239 | [+0.3987, +0.4492] | 0.0002 | yes | yes |
| ET vs HierBayes | -0.1059 | [-0.1317, -0.0801] | 0.0002 | yes | yes |
| ET vs GLM-Logit | -0.1286 | [-0.1573, -0.0997] | 0.0002 | yes | yes |
| ET vs Poisson-GLM | +0.3225 | [+0.2981, +0.3470] | 0.0002 | yes | yes |
| B4: B1+B3 vs Poisson-GLM | +0.4150 | [+0.3906, +0.4400] | 0.0002 | yes | yes |
| B4: B1+B3 vs ET | +0.0925 | [+0.0672, +0.1178] | 0.0002 | yes | yes |
| B3: DoY season vs Poisson-GLM | +0.3488 | [+0.3265, +0.3719] | 0.0002 | yes | yes |
| ET vs TTM | -0.0692 | [-0.0952, -0.0413] | 0.0002 | yes | yes |
| B1: Static prior vs B3: DoY season | +0.0526 | [+0.0256, +0.0807] | 0.0004 | yes | yes |
| B2: Recent MA vs B3: DoY season | +0.0654 | [+0.0337, +0.0965] | 0.0004 | yes | yes |
| B4: B1+B3 vs GLM-Logit | -0.0360 | [-0.0588, -0.0143] | 0.0012 | no | yes |
| GLM-Logit vs HierBayes | +0.0227 | [+0.0074, +0.0390] | 0.0026 | no | yes |
| B1: Static prior vs HierBayes | -0.0269 | [-0.0455, -0.0093] | 0.0028 | no | yes |
| B3: DoY season vs TTM | -0.0429 | [-0.0727, -0.0148] | 0.0036 | no | yes |
| B2: Recent MA vs GLM-Logit | -0.0368 | [-0.0649, -0.0108] | 0.0086 | no | no |
| B4: B1+B3 vs TTM | +0.0234 | [+0.0024, +0.0443] | 0.0284 | no | no |
| B5: B2+B3 vs TTM | +0.0322 | [+0.0021, +0.0615] | 0.0362 | no | no |
| B1: Static prior vs B4: B1+B3 | -0.0136 | [-0.0274, -0.0001] | 0.0428 | no | no |
| B5: B2+B3 vs GLM-Logit | -0.0271 | [-0.0554, -0.0012] | 0.0482 | no | no |
| B3: DoY season vs ET | +0.0263 | [-0.0030, +0.0540] | 0.0690 | no | no |
| B2: Recent MA vs TTM | +0.0225 | [-0.0080, +0.0515] | 0.1412 | no | no |
| B1: Static prior vs B5: B2+B3 | -0.0225 | [-0.0527, +0.0095] | 0.1614 | no | no |
| B4: B1+B3 vs HierBayes | -0.0133 | [-0.0335, +0.0068] | 0.1956 | no | no |
| B2: Recent MA vs B5: B2+B3 | -0.0097 | [-0.0266, +0.0081] | 0.2701 | no | no |
| B2: Recent MA vs HierBayes | -0.0141 | [-0.0420, +0.0125] | 0.3023 | no | no |
| B1: Static prior vs TTM | +0.0097 | [-0.0109, +0.0305] | 0.3643 | no | no |
| B1: Static prior vs B2: Recent MA | -0.0128 | [-0.0442, +0.0197] | 0.4309 | no | no |
| B4: B1+B3 vs B5: B2+B3 | -0.0089 | [-0.0376, +0.0208] | 0.5645 | no | no |
| B5: B2+B3 vs HierBayes | -0.0044 | [-0.0321, +0.0213] | 0.7429 | no | no |
| B2: Recent MA vs B4: B1+B3 | -0.0008 | [-0.0331, +0.0291] | 0.9566 | no | no |

Generated by `python scripts/table2_significance.py --all`.