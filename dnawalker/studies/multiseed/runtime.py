"""Model-specific runtime descriptions for multi-seed retraining."""

import configparser
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .constants import MODEL_NAMES
from dnawalker.shared.artifacts import optional_checkpoint_seed


def normalize_model(model):
    """Return the canonical model name or raise for an unsupported selector."""
    key = str(model).strip().lower()
    if key not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model {model!r}; expected 'cnn' or 'transformer'."
        )
    return key


@dataclass(frozen=True)
class ModelRunSpec:
    """Filesystem and command contract for one inverse-model trainer."""

    name: str
    train_dir: Path
    trainer_script: str
    checkpoint_template: str
    artifact_dir: Path | None = None
    scaler_template: str | None = None

    def __post_init__(self):
        if self.name not in MODEL_NAMES:
            raise ValueError(
                f"Unsupported model run spec {self.name!r}; "
                f"expected one of {MODEL_NAMES}"
            )
        artifact_dir = (
            self.train_dir / "artifacts" / "models" / self.name
            if self.artifact_dir is None
            else Path(self.artifact_dir)
        )
        object.__setattr__(self, "artifact_dir", artifact_dir.resolve())
        if self.scaler_template is None:
            scaler_template = (
                "y_scaler.seed{seed}.pkl"
                if self.name == "cnn"
                else "transformer_y_scaler.seed{seed}.pkl"
            )
            object.__setattr__(self, "scaler_template", scaler_template)

    def checkpoint_path(self, seed):
        """Return the absolute per-seed checkpoint path."""
        return str(
            self.artifact_dir / self.checkpoint_template.format(seed=seed)
        )

    def scaler_path(self, seed):
        """Return the absolute paired per-seed Y-scaler path."""
        return str(
            self.artifact_dir / self.scaler_template.format(seed=seed)
        )

    def override_sections(
        self,
        seed,
        split_seed,
        dataset_path=None,
        split_manifest_path=None,
        train_subset_size=None,
    ):
        """Return the complete INI override for one isolated seeded run."""
        training = {
            "random_seed": str(seed),
            "split_seed": str(split_seed),
        }
        if dataset_path is not None:
            dataset_path = str(Path(dataset_path).resolve())
        if (split_manifest_path is None) != (train_subset_size is None):
            raise ValueError(
                "split_manifest_path and train_subset_size must be provided together"
            )
        if split_manifest_path is not None:
            training.update({
                "split_manifest_file": str(
                    Path(split_manifest_path).resolve()
                ),
                "train_subset_size": str(int(train_subset_size)),
            })

        if self.name == "cnn":
            training.update({
                "model_save_path": self.checkpoint_path(seed),
                "y_scaler_file": self.scaler_path(seed),
            })
            sections = {"TRAINING": training}
            if dataset_path is not None:
                sections["DATA_GENERATION"] = {
                    "output_filename": dataset_path,
                }
            return sections

        sections = {
            "TRAINING": training,
            "TRANSFORMER": {
                "model_save_path": self.checkpoint_path(seed),
                "y_scaler_file": self.scaler_path(seed),
            },
        }
        if dataset_path is not None:
            # The same override is consumed by the parent Config and the
            # TransformerConfig, so both dataset keys must point at the exact
            # release artifact.
            sections["DATA_GENERATION"] = {
                "output_filename": dataset_path,
            }
            sections["PATHS"] = {
                "dataset_file": dataset_path,
                "model_artifacts_dir": str(self.artifact_dir),
            }
        return sections

    def training_command(self, python_executable, override_path):
        """Build the isolated ``python -m`` trainer command."""
        command = [
            python_executable,
            "-m",
            self.trainer_script,
            "--config",
            override_path,
        ]
        if self.name == "transformer":
            command.extend(["--transformer-config", override_path])
        return command


@dataclass(frozen=True)
class TrainOutcome:
    """Result of one seed's training subprocess."""

    seed: int
    model_path: str | None
    ok: bool
    error: str | None
    log_path: str | None = None


def build_model_specs(repo_root, artifact_root=None):
    """Build runtime specs under an explicit, isolated artifact namespace."""
    root = Path(repo_root).resolve()
    model_root = (
        root / "artifacts" / "models"
        if artifact_root is None
        else Path(artifact_root).resolve()
    )
    return {
        "cnn": ModelRunSpec(
            name="cnn",
            train_dir=root,
            artifact_dir=model_root / "cnn",
            trainer_script="dnawalker.cnn.train",
            checkpoint_template="best_mlp_model.seed{seed}.pth",
            scaler_template="y_scaler.seed{seed}.pkl",
        ),
        "transformer": ModelRunSpec(
            name="transformer",
            train_dir=root,
            artifact_dir=model_root / "transformer",
            trainer_script="dnawalker.transformer.train",
            checkpoint_template="best_transformer_model.seed{seed}.pth",
            scaler_template="transformer_y_scaler.seed{seed}.pkl",
        ),
    }


def write_override_file(
        spec, seed, split_seed, override_dir, dataset_path=None,
        split_manifest_path=None, train_subset_size=None):
    """Write one model's complete per-seed override and return its path."""
    output_dir = Path(override_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = configparser.ConfigParser()
    parser.optionxform = str
    sections = spec.override_sections(
        seed,
        split_seed,
        dataset_path=dataset_path,
        split_manifest_path=split_manifest_path,
        train_subset_size=train_subset_size,
    )
    for section, values in sections.items():
        parser[section] = values

    output_path = output_dir / f"{spec.name}_seed{seed}.ini"
    with output_path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return str(output_path.resolve())


def stderr_tail(text, limit=2000):
    """Return the trailing ``limit`` characters of captured stderr."""
    if not text:
        return None
    stripped = text.strip()
    return stripped[-limit:] if len(stripped) > limit else stripped


def _write_training_log(log_path, command, result):
    """Persist one trainer's command, stdout, and stderr for auditability."""
    if log_path is None:
        return None
    path = Path(log_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND\n")
        handle.write(" ".join(str(part) for part in command))
        handle.write("\n\nSTDOUT\n")
        handle.write(stdout)
        handle.write("\n\nSTDERR\n")
        handle.write(stderr)
        handle.write(f"\n\nEXIT_CODE\n{result.returncode}\n")
    return str(path)


def execute_training(
        spec, seed, override_path, python_executable, runner=None,
        log_path=None):
    """Execute one trainer and convert expected failures to ``TrainOutcome``."""
    runner = subprocess.run if runner is None else runner
    model_path = spec.checkpoint_path(seed)
    command = spec.training_command(python_executable, override_path)
    try:
        result = runner(
            command,
            cwd=str(spec.train_dir),
            capture_output=True,
            text=True,
        )
        saved_log = _write_training_log(log_path, command, result)
        if result.returncode != 0:
            error_output = (
                getattr(result, "stderr", None)
                or getattr(result, "stdout", None)
            )
            return TrainOutcome(
                seed=seed,
                model_path=None,
                ok=False,
                error=(
                    f"trainer exited with code {result.returncode}: "
                    f"{stderr_tail(error_output) or '<no output>'}"
                ),
                log_path=saved_log,
            )
        if not os.path.exists(model_path):
            return TrainOutcome(
                seed=seed,
                model_path=None,
                ok=False,
                error=f"checkpoint not found after training: {model_path}",
                log_path=saved_log,
            )
        return TrainOutcome(
            seed=seed,
            model_path=model_path,
            ok=True,
            error=None,
            log_path=saved_log,
        )
    except Exception as exc:
        return TrainOutcome(
            seed=seed,
            model_path=None,
            ok=False,
            error=repr(exc),
            log_path=(str(Path(log_path).resolve()) if log_path else None),
        )


def checkpoint_split_seed(path):
    """Read split provenance from a safe tensor checkpoint, or return ``None``."""
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # A stale, truncated, or foreign checkpoint is an unusable seed, not a
        # reason to abort the remaining multi-seed sweep.
        return None
    if not isinstance(checkpoint, dict) or "split_seed" not in checkpoint:
        return None
    try:
        return optional_checkpoint_seed(
            checkpoint["split_seed"], "split_seed"
        )
    except ValueError:
        return None
