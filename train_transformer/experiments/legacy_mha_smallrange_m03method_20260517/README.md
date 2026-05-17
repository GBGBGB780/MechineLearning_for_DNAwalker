# Legacy MHA Small-Range m03-Method Round

This round uses the exact m03 baseline training method on previously generated small-range datasets.

Definition of exact m03 baseline method:
- legacy MHA transformer
- weighted training from scratch (adaptive weighted loss)
- no checkpoint initialization from m03 or m17
- same baseline hyperparameters as current_baseline_m03

Variants:
- `s15_m03method_origknn_1024`: Exact m03 weighted-from-scratch training on the orig-centered KNN subset (1024).
- `s16_m03method_dualunion_2048`: Exact m03 weighted-from-scratch training on the orig/gen union subset (2048).
- `s17_m03method_corridor_2048`: Exact m03 weighted-from-scratch training on the orig/gen corridor subset (2048).
- `s18_m03method_origwin30_352`: Exact m03 weighted-from-scratch training on the true 30% orig-centered window subset (352).
- `s19_m03method_origwin35_861`: Exact m03 weighted-from-scratch training on the true 35% orig-centered window subset (861).
