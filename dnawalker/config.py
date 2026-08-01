# coding=utf-8
"""Shared configuration access for ``configs/common.ini``.

The shared loader owns physics, data, split, preprocessing, and prediction
contracts. Model-specific settings are layered by ``CNNConfig`` and
``TransformerConfig`` in their respective packages.
"""

import configparser
import math
import os

from dnawalker.paths import ARTIFACTS_DIR, DEFAULT_CONFIG, RESULTS_DIR


def _normalize_config_files(config_file, extra_config_files=None):
    """Return an ordered list of config files with existence checks."""
    config_files = [config_file]
    if extra_config_files:
        if isinstance(extra_config_files, (str, os.PathLike)):
            config_files.append(extra_config_files)
        else:
            config_files.extend(extra_config_files)

    normalized = []
    for path in config_files:
        if path is None:
            continue
        path = os.fspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在 / Config not found: {path}")
        normalized.append(path)
    return normalized


class Config:
    """Unified access to the shared DNA Walker configuration."""

    def __init__(self, config_file=None, extra_config_files=None,
                 validate=True):
        """
        Initialize the configuration loader.

        Args:
            config_file: Primary INI path. ``None`` uses ``configs/common.ini``
                independently of the caller's working directory.
            extra_config_files: Optional override file or ordered sequence.
            validate: Validate required keys and ranges after loading.
        """
        if config_file is None:
            config_file = os.fspath(DEFAULT_CONFIG)
        config_files = _normalize_config_files(config_file, extra_config_files)
        self.config = configparser.ConfigParser()
        self.config.read(config_files, encoding='utf-8')
        self.config_file = config_files[0]
        self.config_files = config_files
        self._config_dir = os.path.dirname(os.path.abspath(self.config_file))

        if validate:
            self._validate()

    def _validate(self):
        """在运行前校验跨模块契约和会导致静默错误的数值范围。

        Validate shared model/data contracts and numeric ranges before an
        expensive generation, training, or inference run starts.
        """
        required = [
            ('TRAINING', 'sim_duration_minutes'),
            ('TRAINING', 'num_curves'),
            ('TRAINING', 'output_size'),
            ('TRAINING', 'random_seed'),
        ]
        missing = [f"[{s}].{k}" for s, k in required
                   if not self.config.has_option(s, k)]
        if missing:
            raise ValueError(
                f"配置缺少必需项 / Config missing required keys: {missing}"
            )

        # 物理合理性 / Sanity ranges on training hyperparams.
        duration = self.config.getint('TRAINING', 'sim_duration_minutes')
        curves = self.config.getint('TRAINING', 'num_curves')
        output_size = self.config.getint('TRAINING', 'output_size')
        if duration <= 0:
            raise ValueError(
                "sim_duration_minutes 必须 > 0 / must be positive: "
                f"got {duration}"
            )
        if curves <= 0:
            raise ValueError(
                f"num_curves 必须 > 0 / must be positive: got {curves}"
            )
        if curves != 3:
            raise ValueError(
                "num_curves 必须为 3 (FAM/TYE/CY5) / must be exactly 3: "
                f"got {curves}"
            )
        if output_size <= 0:
            raise ValueError(
                f"output_size 必须 > 0 / must be positive: got {output_size}"
            )

        for seed_name in ('random_seed', 'split_seed'):
            if self.config.has_option('TRAINING', seed_name):
                seed = self.config.getint('TRAINING', seed_name)
                if not 0 <= seed <= 2 ** 32 - 1:
                    raise ValueError(
                        f"{seed_name} must be in [0, {2 ** 32 - 1}], got {seed}"
                    )

        test = self.config.getfloat('TRAINING', 'test_split_ratio', fallback=0.1)
        val = self.config.getfloat('TRAINING', 'val_split_ratio', fallback=0.1)
        if (not math.isfinite(test) or not math.isfinite(val)
                or not (0 < test < 1) or not (0 < val < 1)):
            raise ValueError(
                f"test/val split 比例必须在 (0,1) / split ratios out of range: "
                f"test={test}, val={val}"
            )

        split_manifest = self.get_split_manifest_file()
        train_subset_size = self.get_train_subset_size()
        if (split_manifest is None) != (train_subset_size is None):
            raise ValueError(
                "split_manifest_file 与 train_subset_size 必须同时配置 / "
                "must be configured together"
            )
        if train_subset_size is not None and train_subset_size <= 0:
            raise ValueError(
                "train_subset_size 必须 > 0 / must be positive: "
                f"got {train_subset_size}"
            )

        if self.config.has_section('PHYSICAL_PARAMETERS'):
            trainable = self.get_trainable_param_names()
            if output_size != len(trainable):
                raise ValueError(
                    "output_size 与待训练参数数量不一致 / does not match "
                    f"trainable parameters: {output_size} vs {len(trainable)}"
                )
            ranges = self.get_param_ranges()
            missing_ranges = [name for name in trainable if name not in ranges]
            if missing_ranges:
                raise ValueError(
                    "配置缺少待训练参数范围 / missing ranges for trainable "
                    f"parameters: {missing_ranges}"
                )
            log_params = {
                name.strip().lower()
                for name in self.config.get(
                    'DATA_PROCESSING',
                    'log_transform_params',
                    fallback='',
                ).split(',')
                if name.strip()
            }
            unknown_log = log_params - {name.lower() for name in trainable}
            if unknown_log:
                raise ValueError(
                    "log_transform_params 含未知参数 / contains unknown "
                    f"parameters: {sorted(unknown_log)}"
                )
            for name in trainable:
                low, _ = ranges[name]
                if name.lower() in log_params and low <= 0:
                    raise ValueError(
                        f"对数变换参数 {name} 的下界必须 > 0 / log-transformed "
                        f"parameter must be positive: {low}"
                )

        loss_weight_mode = self.get_loss_weight_mode().lower()
        if loss_weight_mode not in {'adaptive', 'none'}:
            raise ValueError(
                "loss_weight_mode 必须是 adaptive 或 none / must be one of "
                f"adaptive, none: {loss_weight_mode!r}"
            )

        if self.config.has_section('DATA_PROCESSING'):
            safe_threshold = self.get_safe_threshold()
            nan_replacement = self.get_nan_replacement_value()
            log_epsilon = self.get_log_epsilon()
            if not math.isfinite(safe_threshold) or safe_threshold <= 0:
                raise ValueError(
                    f"safe_threshold must be finite and positive: {safe_threshold}"
                )
            if (not math.isfinite(nan_replacement)
                    or abs(nan_replacement) < safe_threshold):
                raise ValueError(
                    "nan_replacement_value must be finite and outside the "
                    f"accepted safe range: {nan_replacement}"
                )
            if not math.isfinite(log_epsilon) or log_epsilon <= 0:
                raise ValueError(
                    f"log_epsilon must be finite and positive: {log_epsilon}"
                )
            if self.get_amplitude_filter_enabled():
                thresholds = self.get_amplitude_thresholds()
                if any(not math.isfinite(x) or x < 0 for x in thresholds):
                    raise ValueError(
                        "amplitude thresholds must be finite and non-negative: "
                        f"{thresholds}"
                    )

        edges = self.get_bin_edges()
        if (len(edges) < 2
                or any(not math.isfinite(value) for value in edges)
                or any(right <= left for left, right in zip(edges, edges[1:]))):
            raise ValueError(
                "bin_edges 必须是至少两个严格递增的有限值 / must contain "
                f"at least two strictly increasing finite values: {edges}"
            )
        weak_floor = self.get_weak_signal_floor()
        if not math.isfinite(weak_floor) or weak_floor < 0:
            raise ValueError(
                "weak_signal_floor 必须是有限非负数 / must be finite and "
                f"non-negative: {weak_floor}"
            )

        if self.config.has_section('NANOROBOT_MODELING'):
            sim_total = self.get_sim_total_time()
            if not math.isfinite(sim_total) or sim_total <= 0:
                raise ValueError(
                    "sim_total_time 必须是有限正数 / must be finite and "
                    f"positive: {sim_total}"
                )
            if not math.isclose(sim_total, duration, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    "sim_total_time 必须与 sim_duration_minutes 一致 / must "
                    f"match: {sim_total} vs {duration}"
                )

        if self.config.has_section('PREDICTION'):
            window = self.get_sg_window()
            order = self.get_sg_polyorder()
            if (window <= 0 or window % 2 == 0 or window > self.get_seq_length()
                    or order < 0 or order >= window):
                raise ValueError(
                    "SG 配置无效 / invalid Savitzky-Golay settings: "
                    f"window={window}, polyorder={order}"
                )
            ensemble_size = self.get_ensemble_size()
            noise = self.get_ensemble_noise_std()
            if ensemble_size <= 0:
                raise ValueError(
                    f"ensemble_size 必须 > 0 / must be positive: {ensemble_size}"
                )
            if not math.isfinite(noise) or noise < 0:
                raise ValueError(
                    "ensemble_noise_std 必须是有限非负数 / must be finite and "
                    f"non-negative: {noise}"
                )

    def _resolve_from_config_dir(self, path):
        """将配置中的路径解析为绝对路径。/ Resolve config path against config dir."""
        if path.startswith('./'):
            path = path[2:]
        if not os.path.isabs(path):
            path = os.path.join(self._config_dir, path)
        return os.path.normpath(path)

    @staticmethod
    def _to_cwd_relative(path):
        """将绝对路径转换为相对当前工作目录的路径。/ Convert to cwd-relative path."""
        return os.path.relpath(os.path.normpath(path), start=os.getcwd())

    # ==================== TRAINING 参数 / Training Parameters ====================

    def get_sim_duration_minutes(self):
        """获取模拟持续时间（分钟）/ Get simulation duration in minutes."""
        return self.config.getint('TRAINING', 'sim_duration_minutes')

    def get_num_curves(self):
        """获取输入曲线数量 (3: FAM, TYE, CY5) / Get number of input curves."""
        return self.config.getint('TRAINING', 'num_curves')

    def get_seq_length(self):
        """
        获取时间序列长度（自动计算）/ Get sequence length (auto-computed).
        公式 / Formula: sim_duration_minutes × 60 + 1
        """
        return self.get_sim_duration_minutes() * 60 + 1

    def get_input_size(self):
        """获取输入维度 / Get input dimension: num_curves × seq_length."""
        return self.get_num_curves() * self.get_seq_length()

    def get_output_size(self):
        """获取输出维度（待预测参数数量）/ Get output dimension (parameter count)."""
        return self.config.getint('TRAINING', 'output_size')

    def get_output_path(self):
        """Get the repository/config-owned generated-results directory."""
        path = self.config.get(
            'PATHS',
            'output_path',
            fallback=os.fspath(RESULTS_DIR),
        )
        return self._resolve_from_config_dir(path)

    def get_artifacts_path(self):
        """Get the external-artifact root, anchored to the primary config."""
        path = self.config.get(
            'PATHS',
            'artifacts_path',
            fallback=os.fspath(ARTIFACTS_DIR),
        )
        return self._resolve_from_config_dir(path)

    def get_dataset_dir(self):
        """Get the external dataset directory, independent of caller cwd."""
        path = self.config.get(
            'PATHS',
            'dataset_dir',
            fallback=os.fspath(ARTIFACTS_DIR / 'datasets'),
        )
        return self._resolve_from_config_dir(path)

    # 数据集拆分 / Dataset split
    def get_test_split_ratio(self):
        """获取测试集比例 / Get test split ratio."""
        return self.config.getfloat('TRAINING', 'test_split_ratio')

    def get_val_split_ratio(self):
        """获取验证集比例 / Get validation split ratio."""
        return self.config.getfloat('TRAINING', 'val_split_ratio')

    def get_random_seed(self):
        """获取模型训练随机种子 / Get the model-training random seed."""
        return self.config.getint('TRAINING', 'random_seed')

    def get_split_seed(self):
        """获取固定数据划分种子 / Get the fixed dataset-split seed.

        ``random_seed`` controls model initialization, shuffling, dropout, and
        other training randomness.  Dataset membership must stay fixed across
        model seeds, otherwise a fixed evaluation set can overlap another
        seed's training set.  Older configs without ``split_seed`` explicitly
        fall back to ``random_seed`` for backward compatibility.
        """
        return self.config.getint(
            'TRAINING', 'split_seed', fallback=self.get_random_seed()
        )

    def get_split_manifest_file(self):
        """Return the optional explicit split manifest path."""
        value = self.config.get(
            'TRAINING', 'split_manifest_file', fallback=''
        ).strip()
        if not value:
            return None
        if os.path.isabs(value):
            return os.path.normpath(value)
        return self._resolve_from_config_dir(value)

    def get_train_subset_size(self):
        """Return the optional explicit learning-curve training size."""
        value = self.config.get(
            'TRAINING', 'train_subset_size', fallback=''
        ).strip()
        return int(value) if value else None

    def get_loss_weight_mode(self):
        """获取损失函数权重模式 / Get loss weight mode."""
        return self.config.get('TRAINING', 'loss_weight_mode', fallback='none').strip()

    # ==================== DATA_PROCESSING 参数 / Data Processing ====================

    def get_safe_threshold(self):
        """获取数据清洗安全阈值 / Get data cleanup safety threshold."""
        return self.config.getfloat('DATA_PROCESSING', 'safe_threshold')

    def get_nan_replacement_value(self):
        """获取 NaN 替换值 / Get NaN replacement value."""
        return self.config.getfloat('DATA_PROCESSING', 'nan_replacement_value')

    def get_log_epsilon(self):
        """获取 log 变换保护值 / Get log transform epsilon."""
        return self.config.getfloat('DATA_PROCESSING', 'log_epsilon')

    def get_log_transform_params(self):
        """获取需要 log10 变换的参数列表 / Get parameter names requiring log10 transform."""
        raw = self.config.get('DATA_PROCESSING', 'log_transform_params', fallback='')
        return [p.strip() for p in raw.split(',') if p.strip()]

    # 振幅过滤 / Amplitude filtering
    def get_amplitude_filter_enabled(self):
        """是否启用振幅过滤 / Whether amplitude filtering is enabled."""
        return self.config.getboolean('DATA_PROCESSING', 'amplitude_filter_enabled', fallback=False)

    def get_amplitude_thresholds(self):
        """获取各通道振幅阈值 [FAM, TYE, CY5] / Get per-channel amplitude thresholds."""
        return [
            self.config.getfloat('DATA_PROCESSING', 'amplitude_max_fam', fallback=1.0),
            self.config.getfloat('DATA_PROCESSING', 'amplitude_max_tye', fallback=1.0),
            self.config.getfloat('DATA_PROCESSING', 'amplitude_max_cy5', fallback=1.0),
        ]

    # ==================== DATA_GENERATION 参数 / Data Generation ====================

    def get_num_time_points(self):
        """获取时间点数量（等于 seq_length）/ Get number of time points (equals seq_length)."""
        return self.get_seq_length()

    def get_dataset_file(self):
        """Get the external dataset path, independent of caller cwd."""
        filename = self.config.get('DATA_GENERATION', 'output_filename')
        if os.path.isabs(filename):
            return os.path.normpath(filename)
        return os.path.normpath(os.path.join(self.get_dataset_dir(), filename))

    def get_weak_signal_floor(self):
        """获取弱信号过滤阈值 / Weak-signal floor used by dnawalker.data.generate."""
        return self.config.getfloat('DATA_GENERATION', 'weak_signal_floor', fallback=0.01)

    def get_bin_edges(self):
        """获取活跃度分箱边界 / Get the activity-bin edges for balanced sampling.

        Returns:
            list[float]: edges parsed from the config (fallback matches gendata.m).
        """
        raw = self.config.get('DATA_GENERATION', 'bin_edges',
                              fallback='0.01, 0.05, 0.1, 0.2, 0.4, 2.0')
        return [float(x.strip()) for x in raw.split(',') if x.strip()]

    # ==================== PHYSICAL_PARAMETERS 物理参数 ====================

    def get_trainable_param_names(self):
        """
        获取待训练参数名称（值为空的参数）。
        Get trainable parameter names (parameters with empty values).

        Returns:
            list: 参数名称列表 / list of parameter names
        """
        params = []
        for key in self.config['PHYSICAL_PARAMETERS']:
            value = self.config['PHYSICAL_PARAMETERS'][key].strip()
            if value == '' or value.startswith('#'):
                params.append(key)
        return params

    def get_p_unbind_track(self):
        """获取轨道解绑概率 / Get track unbinding probability."""
        return self.config.getfloat('PHYSICAL_PARAMETERS', 'p_unbind_track')

    # ==================== TRAINING_PARAMETER_RANGES 参数范围 ====================

    def get_param_ranges(self):
        """
        获取待训练参数的 [min, max] 范围。
        Get [min, max] ranges for trainable parameters.

        Returns:
            dict: {param_name: (min_val, max_val)}
        """
        def parse_range(name, value):
            try:
                parts = [float(item.strip()) for item in value.split(',')]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"参数范围格式无效 / invalid range for {name}: {value!r}"
                ) from exc
            if (len(parts) != 2
                    or not all(math.isfinite(item) for item in parts)
                    or parts[0] >= parts[1]):
                raise ValueError(
                    f"参数范围必须为有限且 min < max / invalid range for "
                    f"{name}: {value!r}"
                )
            return tuple(parts)

        default_str = self.config.get(
            'TRAINING_PARAMETER_RANGES',
            'default_range',
            fallback='-3.0, 3.0',
        )
        default_range = parse_range('default_range', default_str)

        ranges = {}
        for name in self.get_trainable_param_names():
            range_str = self.config.get('TRAINING_PARAMETER_RANGES', name, fallback=None)
            if range_str:
                ranges[name] = parse_range(name, range_str)
            else:
                ranges[name] = default_range
        return ranges

    # ==================== NANOROBOT_MODELING 纳米机器人建模 ====================

    def get_experimental_data_path(self):
        """Get the experimental-input path, anchored to the primary config."""
        path = self.config.get('NANOROBOT_MODELING', 'path_to_experimental_data_a')
        if os.path.isabs(path):
            return os.path.normpath(path)
        if path.startswith('./'):
            path = path[2:]
        if os.path.dirname(path):
            return self._resolve_from_config_dir(path)
        # Legacy configs used a bare filename located under output_path.
        return os.path.normpath(os.path.join(self.get_output_path(), path))

    def get_sim_total_time(self):
        """获取模拟总时长 / Get total simulation time."""
        return self.config.getfloat('NANOROBOT_MODELING', 'sim_total_time')

    # ==================== PREDICTION 预测参数 ====================

    def get_sg_window(self):
        """获取 SG 平滑窗口大小 / Get Savitzky-Golay window size."""
        return self.config.getint('PREDICTION', 'sg_window', fallback=61)

    def get_sg_polyorder(self):
        """获取 SG 多项式阶数 / Get Savitzky-Golay polynomial order."""
        return self.config.getint('PREDICTION', 'sg_polyorder', fallback=3)

    def get_ensemble_size(self):
        """获取集成预测次数 / Get ensemble prediction count."""
        return self.config.getint('PREDICTION', 'ensemble_size', fallback=50)

    def get_ensemble_noise_std(self):
        """获取集成扰动标准差 / Get ensemble perturbation std."""
        return self.config.getfloat('PREDICTION', 'ensemble_noise_std', fallback=0.005)

    def get_nm_lower_bound(self):
        """获取 Nelder-Mead 下界 / Get Nelder-Mead lower bound."""
        return self.config.getfloat('PREDICTION', 'nm_lower_bound', fallback=0.05)

    def get_nm_upper_bound(self):
        """获取 Nelder-Mead 上界 / Get Nelder-Mead upper bound."""
        return self.config.getfloat('PREDICTION', 'nm_upper_bound', fallback=0.95)



if __name__ == "__main__":
    # 配置加载测试 / Config loading test
    try:
        config = Config()
        print("=== 配置加载测试 / Config Loading Test ===")
        print(f"Input size:  {config.get_input_size()}")
        print(f"Output size: {config.get_output_size()}")
        print(f"Params:      {config.get_trainable_param_names()}")
        print(f"SG window:   {config.get_sg_window()}")
        print(f"Ensemble:    {config.get_ensemble_size()}")
        print("\n配置加载成功 / Config loaded successfully!")
    except Exception as e:
        print(f"配置加载失败 / Config load failed: {e}")
