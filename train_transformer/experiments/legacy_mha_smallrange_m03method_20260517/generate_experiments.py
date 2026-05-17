import json
from pathlib import Path


EXP_NAME = "legacy_mha_smallrange_m03method_20260517"
REMOTE_PROJECT_ROOT = "/oceanstor/home/e1554355/MechineLearning_for_DNAwalker"
REMOTE_EXP_ROOT = f"{REMOTE_PROJECT_ROOT}/train_transformer/experiments/{EXP_NAME}"
LOCAL_EXP_ROOT = Path(__file__).resolve().parent

SMALL_DATA_ROOT = f"{REMOTE_PROJECT_ROOT}/train_transformer/experiments/legacy_mha_smallrange_20260516/datasets"


VARIANTS = [
    {
        "tag": "s15_m03method_origknn_1024",
        "job_name": "s15m03",
        "dataset": "training_dataset_small_origknn_1024.npz",
        "note": "Exact m03 weighted-from-scratch training on the orig-centered KNN subset (1024).",
    },
    {
        "tag": "s16_m03method_dualunion_2048",
        "job_name": "s16m03",
        "dataset": "training_dataset_small_dualunion_2048.npz",
        "note": "Exact m03 weighted-from-scratch training on the orig/gen union subset (2048).",
    },
    {
        "tag": "s17_m03method_corridor_2048",
        "job_name": "s17m03",
        "dataset": "training_dataset_small_corridor_2048.npz",
        "note": "Exact m03 weighted-from-scratch training on the orig/gen corridor subset (2048).",
    },
    {
        "tag": "s18_m03method_origwin30_352",
        "job_name": "s18m03",
        "dataset": "training_dataset_small_origwin30_actual.npz",
        "note": "Exact m03 weighted-from-scratch training on the true 30% orig-centered window subset (352).",
    },
    {
        "tag": "s19_m03method_origwin35_861",
        "job_name": "s19m03",
        "dataset": "training_dataset_small_origwin35_actual.npz",
        "note": "Exact m03 weighted-from-scratch training on the true 35% orig-centered window subset (861).",
    },
]


def ensure_dirs():
    for rel in ["configs", "jobs", "logs", "outputs"]:
        (LOCAL_EXP_ROOT / rel).mkdir(parents=True, exist_ok=True)


def config_text(variant):
    dataset_path = f"{SMALL_DATA_ROOT}/{variant['dataset']}"
    output_dir = f"{REMOTE_EXP_ROOT}/outputs/{variant['tag']}"
    model_path = f"{output_dir}/best_transformer_model.{variant['tag']}.pth"
    scaler_path = f"{output_dir}/transformer_y_scaler.{variant['tag']}.pkl"
    return (
        "[TRANSFORMER]\n"
        "patch_size = 50\n"
        "stride = 25\n"
        "d_model = 256\n"
        "n_heads = 8\n"
        "n_layers = 4\n"
        "d_ff = 512\n"
        "cross_channel_layers = 2\n"
        "dropout = 0.15\n"
        "dropout_head = 0.30\n"
        "learning_rate = 1.0e-4\n"
        "weight_decay = 1.0e-4\n"
        "warmup_ratio = 0.1\n"
        "scheduler_type = cosine\n"
        "scheduler_min_lr = 1e-6\n"
        "batch_size = 32\n"
        "num_epochs = 1000\n"
        "early_stopping_patience = 150\n"
        f"model_save_path = {model_path}\n"
        f"y_scaler_file = {scaler_path}\n"
        "\n"
        "[PATHS]\n"
        f"dataset_file = {dataset_path}\n"
    )


def job_text(variant):
    config_path = f"{REMOTE_EXP_ROOT}/configs/config_transformer.{variant['tag']}.ini"
    output_dir = f"{REMOTE_EXP_ROOT}/outputs/{variant['tag']}"
    model_path = f"{output_dir}/best_transformer_model.{variant['tag']}.pth"
    scaler_path = f"{output_dir}/transformer_y_scaler.{variant['tag']}.pkl"
    return f"""#!/bin/bash
#PBS -N {variant['job_name']}
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q ais_gpu
#PBS -V
#PBS -o {REMOTE_EXP_ROOT}/logs/{variant['tag']}.out
#PBS -e {REMOTE_EXP_ROOT}/logs/{variant['tag']}.err

set -eo pipefail
source /etc/profile
cd {REMOTE_PROJECT_ROOT}

module load python/3.12.7
module load matlab
module load cuda/12.4

source /home/svu/e1554355/miniconda3/etc/profile.d/conda.sh
conda activate dna_env

mkdir -p {REMOTE_EXP_ROOT}/logs {output_dir}
test -f {SMALL_DATA_ROOT}/{variant['dataset']}

python -u {REMOTE_EXP_ROOT}/train_legacy_mha.py --config-base configfile.ini --transformer-config {config_path} --adaptive-weight

python -u {REMOTE_EXP_ROOT}/evaluate_legacy_mha_rmse.py \\
  --config-base configfile.ini \\
  --config-generalization configfile.generalization.ini \\
  --transformer-config {config_path} \\
  --model {model_path} \\
  --scaler {scaler_path} \\
  --output-dir {output_dir} \\
  --tag {variant['tag']}
"""


def submit_script():
    lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        "source /etc/profile",
        'cd "$(dirname "$0")"',
        "mkdir -p logs outputs",
    ]
    for variant in VARIANTS:
        lines.append(f'jid=$(/opt/pbs/bin/qsub jobs/run_{variant["tag"]}.pbs)')
        lines.append(f'echo "submitted {variant["tag"]}: ${{jid}}"')
    return "\n".join(lines) + "\n"


def readme_text():
    lines = [
        "# Legacy MHA Small-Range m03-Method Round",
        "",
        "This round uses the exact m03 baseline training method on previously generated small-range datasets.",
        "",
        "Definition of exact m03 baseline method:",
        "- legacy MHA transformer",
        "- weighted training from scratch (adaptive weighted loss)",
        "- no checkpoint initialization from m03 or m17",
        "- same baseline hyperparameters as current_baseline_m03",
        "",
        "Variants:",
    ]
    for variant in VARIANTS:
        lines.append(f"- `{variant['tag']}`: {variant['note']}")
    return "\n".join(lines) + "\n"


def main():
    ensure_dirs()
    for variant in VARIANTS:
        (LOCAL_EXP_ROOT / "configs" / f"config_transformer.{variant['tag']}.ini").write_text(
            config_text(variant),
            encoding="utf-8",
        )
        (LOCAL_EXP_ROOT / "jobs" / f"run_{variant['tag']}.pbs").write_text(
            job_text(variant),
            encoding="utf-8",
        )
    (LOCAL_EXP_ROOT / "submit_all.sh").write_text(submit_script(), encoding="utf-8")
    (LOCAL_EXP_ROOT / "README.md").write_text(readme_text(), encoding="utf-8")
    (LOCAL_EXP_ROOT / "variants.json").write_text(
        json.dumps({"variants": VARIANTS, "remote_exp_root": REMOTE_EXP_ROOT}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Generated {EXP_NAME}")


if __name__ == "__main__":
    main()
