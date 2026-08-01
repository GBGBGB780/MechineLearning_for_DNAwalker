"""CNN configuration layered over the shared project configuration."""

import math
import os

from dnawalker.config import Config
from dnawalker.paths import ARTIFACTS_DIR, DEFAULT_CNN_CONFIG, DEFAULT_CONFIG


DEFAULT_ARTIFACT_DIR = os.fspath(ARTIFACTS_DIR / "models" / "cnn")


def resolve_artifact_path(path):
    """Resolve a CNN artifact path independently of the caller's cwd.

    Absolute paths remain explicit overrides. Legacy ``results/<name>``
    values are redirected to the centralized CNN artifact directory.
    """
    path = os.fspath(path)
    if os.path.isabs(path):
        return os.path.normpath(path)

    normalized = os.path.normpath(path)
    parts = normalized.split(os.sep)
    if parts and parts[0] == "results":
        normalized = os.path.join(*parts[1:]) if len(parts) > 1 else ""
    return os.path.normpath(os.path.join(DEFAULT_ARTIFACT_DIR, normalized))


class CNNConfig(Config):
    """Load ``common.ini``, ``cnn.ini``, then optional run overrides."""

    def __init__(
        self,
        config_file=None,
        cnn_config_file=None,
        extra_config_files=None,
        validate=True,
    ):
        common_path = os.fspath(config_file or DEFAULT_CONFIG)
        cnn_path = os.fspath(cnn_config_file or DEFAULT_CNN_CONFIG)
        extras = [cnn_path]
        if extra_config_files:
            if isinstance(extra_config_files, (str, os.PathLike)):
                extras.append(extra_config_files)
            else:
                extras.extend(extra_config_files)
        super().__init__(
            config_file=common_path,
            extra_config_files=extras,
            validate=validate,
        )

    def _validate(self):
        super()._validate()
        required = (
            "learning_rate",
            "batch_size",
            "num_epochs",
            "model_save_path",
            "x_scaler_file",
            "y_scaler_file",
            "scheduler_mode",
            "scheduler_factor",
            "scheduler_patience",
            "scheduler_min_lr",
        )
        missing = [
            f"[TRAINING].{name}"
            for name in required
            if not self.config.has_option("TRAINING", name)
        ]
        architecture_keys = (
            "conv1_out_channels", "conv1_kernel_size", "conv1_stride",
            "conv1_padding", "conv2_out_channels", "conv2_kernel_size",
            "conv2_stride", "conv2_padding", "conv3_out_channels",
            "conv3_kernel_size", "conv3_stride", "conv3_padding",
            "conv4_out_channels", "conv4_kernel_size", "conv4_stride",
            "conv4_padding", "fc1_out_features", "dropout_conv", "dropout_fc",
        )
        missing.extend(
            f"[MODEL_ARCHITECTURE].{name}"
            for name in architecture_keys
            if not self.config.has_option("MODEL_ARCHITECTURE", name)
        )
        if not self.config.has_option("PATHS", "cnn_artifacts_path"):
            missing.append("[PATHS].cnn_artifacts_path")
        if missing:
            raise ValueError(
                f"CNN 配置缺少必需项 / missing required keys: {missing}"
            )

        learning_rate = self.get_learning_rate()
        if not math.isfinite(learning_rate) or not (0 < learning_rate < 1):
            raise ValueError(
                "learning_rate 必须在 (0,1) 范围内 / out of range: "
                f"got {learning_rate}"
            )
        if self.get_batch_size() <= 0:
            raise ValueError("batch_size 必须 > 0 / must be positive")
        if self.get_num_epochs() <= 0:
            raise ValueError("num_epochs 必须 > 0 / must be positive")
        if self.get_early_stopping_patience() < 0:
            raise ValueError(
                "early_stopping_patience 必须 >= 0 / must be non-negative"
            )

        scheduler_mode = self.get_scheduler_mode().strip().lower()
        if scheduler_mode not in {"min", "max"}:
            raise ValueError(
                f"scheduler_mode must be 'min' or 'max', got {scheduler_mode!r}"
            )
        factor = self.get_scheduler_factor()
        if not math.isfinite(factor) or not (0 < factor < 1):
            raise ValueError(
                f"scheduler_factor must be finite and in (0,1), got {factor}"
            )
        if self.get_scheduler_patience() < 0:
            raise ValueError("scheduler_patience must be non-negative")
        min_lr = self.get_scheduler_min_lr()
        if not math.isfinite(min_lr) or not (0 <= min_lr <= learning_rate):
            raise ValueError(
                "scheduler_min_lr must be finite and between 0 and "
                f"learning_rate, got {min_lr}"
            )

        convs = (
            self.get_conv1_params(),
            self.get_conv2_params(),
            self.get_conv3_params(),
            self.get_conv4_params(),
        )
        length = self.get_seq_length()
        for index, conv in enumerate(convs, start=1):
            if (
                conv["out_channels"] <= 0
                or conv["kernel_size"] <= 0
                or conv["stride"] <= 0
                or conv["padding"] < 0
            ):
                raise ValueError(
                    f"conv{index} architecture values are invalid: {conv}"
                )
            length = (
                length + 2 * conv["padding"] - conv["kernel_size"]
            ) // conv["stride"] + 1
            if length <= 0:
                raise ValueError(
                    f"conv{index} collapses sequence length to {length}; "
                    "check kernel_size/stride/padding"
                )
            if index < len(convs):
                length //= 2
                if length <= 0:
                    raise ValueError(
                        f"pool after conv{index} collapses sequence length"
                    )
        if self.get_fc1_out_features() <= 0:
            raise ValueError("fc1_out_features must be positive")
        for name, value in (
            ("dropout_conv", self.get_dropout_conv()),
            ("dropout_fc", self.get_dropout_fc()),
        ):
            if not math.isfinite(value) or not (0 <= value < 1):
                raise ValueError(f"{name} must be in [0,1), got {value}")

    def get_learning_rate(self):
        return self.config.getfloat("TRAINING", "learning_rate")

    def get_batch_size(self):
        return self.config.getint("TRAINING", "batch_size")

    def get_num_epochs(self):
        return self.config.getint("TRAINING", "num_epochs")

    def get_early_stopping_patience(self):
        return self.config.getint(
            "TRAINING", "early_stopping_patience", fallback=0
        )

    def get_scheduler_mode(self):
        return self.config.get("TRAINING", "scheduler_mode")

    def get_scheduler_factor(self):
        return self.config.getfloat("TRAINING", "scheduler_factor")

    def get_scheduler_patience(self):
        return self.config.getint("TRAINING", "scheduler_patience")

    def get_scheduler_min_lr(self):
        return self.config.getfloat("TRAINING", "scheduler_min_lr")

    def get_cnn_artifacts_path(self):
        path = self.config.get("PATHS", "cnn_artifacts_path")
        return self._resolve_from_config_dir(path)

    def _resolve_cnn_artifact(self, path):
        path = os.fspath(path)
        if os.path.isabs(path):
            return os.path.normpath(path)
        normalized = os.path.normpath(path)
        parts = normalized.split(os.sep)
        if parts and parts[0] == "results":
            normalized = os.path.join(*parts[1:]) if len(parts) > 1 else ""
        return os.path.normpath(
            os.path.join(self.get_cnn_artifacts_path(), normalized)
        )

    def get_model_save_path(self):
        return self._resolve_cnn_artifact(
            self.config.get("TRAINING", "model_save_path")
        )

    def get_x_scaler_file(self):
        return self._resolve_cnn_artifact(
            self.config.get("TRAINING", "x_scaler_file")
        )

    def get_y_scaler_file(self):
        return self._resolve_cnn_artifact(
            self.config.get("TRAINING", "y_scaler_file")
        )

    def _conv_params(self, index):
        section = "MODEL_ARCHITECTURE"
        prefix = f"conv{index}_"
        return {
            "out_channels": self.config.getint(
                section, prefix + "out_channels"
            ),
            "kernel_size": self.config.getint(
                section, prefix + "kernel_size"
            ),
            "stride": self.config.getint(section, prefix + "stride"),
            "padding": self.config.getint(section, prefix + "padding"),
        }

    def get_conv1_params(self):
        return self._conv_params(1)

    def get_conv2_params(self):
        return self._conv_params(2)

    def get_conv3_params(self):
        return self._conv_params(3)

    def get_conv4_params(self):
        return self._conv_params(4)

    def get_fc1_out_features(self):
        return self.config.getint(
            "MODEL_ARCHITECTURE", "fc1_out_features"
        )

    def get_dropout_conv(self):
        return self.config.getfloat(
            "MODEL_ARCHITECTURE", "dropout_conv", fallback=0.0
        )

    def get_dropout_fc(self):
        return self.config.getfloat(
            "MODEL_ARCHITECTURE", "dropout_fc", fallback=0.0
        )
