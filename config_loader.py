# coding=utf-8
"""
config_loader.py — 统一配置管理类（读取 configfile.ini）
config_loader.py — Unified configuration manager (reads configfile.ini)

所有物理参数、训练超参数、模型结构参数、数据处理参数均通过此类统一访问。
All physical parameters, training hyperparameters, model architecture, and data
processing parameters are accessed through this class.
"""

import configparser
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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
    """
    统一的配置管理类。/ Unified configuration manager.
    从 configfile.ini 读取所有参数。/ Reads all parameters from configfile.ini.
    """

    def __init__(self, config_file='configfile.ini', extra_config_files=None):
        """
        初始化配置加载器。/ Initialize config loader.

        Args:
            config_file: 配置文件路径 / config file path, default 'configfile.ini'
        """
        config_files = _normalize_config_files(config_file, extra_config_files)
        self.config = configparser.ConfigParser()
        self.config.read(config_files, encoding='utf-8')
        self.config_file = config_files[0]
        self.config_files = config_files
        self._config_dir = os.path.dirname(os.path.abspath(self.config_file))

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

    def get_learning_rate(self):
        """获取初始学习率 / Get initial learning rate."""
        return self.config.getfloat('TRAINING', 'learning_rate')

    def get_batch_size(self):
        """获取批次大小 / Get batch size."""
        return self.config.getint('TRAINING', 'batch_size')

    def get_num_epochs(self):
        """获取训练轮数 / Get number of training epochs."""
        return self.config.getint('TRAINING', 'num_epochs')

    def get_output_path(self):
        """获取公共输出目录（相对当前工作目录）/ Get cwd-relative output directory path."""
        path = self.config.get('PATHS', 'output_path', fallback='results')
        return self._to_cwd_relative(self._resolve_from_config_dir(path))

    def get_model_save_path(self):
        """获取模型保存路径 (执行目录下的 results) / Get model save path."""
        filename = self.config.get('TRAINING', 'model_save_path')
        return os.path.join("results", filename)

    def get_x_scaler_file(self):
        """获取 X 归一化器路径 (执行目录下的 results) / Get X scaler save path."""
        filename = self.config.get('TRAINING', 'x_scaler_file')
        return os.path.join("results", filename)

    def get_y_scaler_file(self):
        """获取 Y 归一化器路径 (执行目录下的 results) / Get Y scaler save path."""
        filename = self.config.get('TRAINING', 'y_scaler_file')
        return os.path.join("results", filename)

    # 学习率调度器 / LR scheduler
    def get_scheduler_mode(self):
        """获取调度器模式 (min/max) / Get scheduler mode."""
        return self.config.get('TRAINING', 'scheduler_mode')

    def get_scheduler_factor(self):
        """获取学习率衰减因子 / Get LR decay factor."""
        return self.config.getfloat('TRAINING', 'scheduler_factor')

    def get_scheduler_patience(self):
        """获取调度器耐心值 / Get scheduler patience."""
        return self.config.getint('TRAINING', 'scheduler_patience')

    def get_scheduler_min_lr(self):
        """获取最小学习率 / Get minimum LR."""
        return self.config.getfloat('TRAINING', 'scheduler_min_lr')

    # 数据集拆分 / Dataset split
    def get_test_split_ratio(self):
        """获取测试集比例 / Get test split ratio."""
        return self.config.getfloat('TRAINING', 'test_split_ratio')

    def get_val_split_ratio(self):
        """获取验证集比例 / Get validation split ratio."""
        return self.config.getfloat('TRAINING', 'val_split_ratio')

    def get_random_seed(self):
        """获取随机种子 / Get random seed."""
        return self.config.getint('TRAINING', 'random_seed')

    def get_early_stopping_patience(self):
        """获取 Early Stopping 耐心值 (0=禁用) / Get early stopping patience (0=disabled)."""
        return self.config.getint('TRAINING', 'early_stopping_patience', fallback=0)

    def get_loss_weight_mode(self):
        """获取损失函数权重模式 / Get loss weight mode."""
        return self.config.get('TRAINING', 'loss_weight_mode', fallback='none').strip()

    # ==================== MODEL_ARCHITECTURE 参数 / Model Architecture ====================

    def get_conv1_params(self):
        """获取第一层卷积参数 / Get conv layer 1 parameters."""
        return {
            'out_channels': self.config.getint('MODEL_ARCHITECTURE', 'conv1_out_channels'),
            'kernel_size': self.config.getint('MODEL_ARCHITECTURE', 'conv1_kernel_size'),
            'stride': self.config.getint('MODEL_ARCHITECTURE', 'conv1_stride'),
            'padding': self.config.getint('MODEL_ARCHITECTURE', 'conv1_padding')
        }

    def get_conv2_params(self):
        """获取第二层卷积参数 / Get conv layer 2 parameters."""
        return {
            'out_channels': self.config.getint('MODEL_ARCHITECTURE', 'conv2_out_channels'),
            'kernel_size': self.config.getint('MODEL_ARCHITECTURE', 'conv2_kernel_size'),
            'stride': self.config.getint('MODEL_ARCHITECTURE', 'conv2_stride'),
            'padding': self.config.getint('MODEL_ARCHITECTURE', 'conv2_padding')
        }

    def get_conv3_params(self):
        """获取第三层卷积参数 / Get conv layer 3 parameters."""
        return {
            'out_channels': self.config.getint('MODEL_ARCHITECTURE', 'conv3_out_channels'),
            'kernel_size': self.config.getint('MODEL_ARCHITECTURE', 'conv3_kernel_size'),
            'stride': self.config.getint('MODEL_ARCHITECTURE', 'conv3_stride'),
            'padding': self.config.getint('MODEL_ARCHITECTURE', 'conv3_padding')
        }

    def get_conv4_params(self):
        """获取第四层卷积参数 / Get conv layer 4 parameters."""
        return {
            'out_channels': self.config.getint('MODEL_ARCHITECTURE', 'conv4_out_channels'),
            'kernel_size': self.config.getint('MODEL_ARCHITECTURE', 'conv4_kernel_size'),
            'stride': self.config.getint('MODEL_ARCHITECTURE', 'conv4_stride'),
            'padding': self.config.getint('MODEL_ARCHITECTURE', 'conv4_padding')
        }

    def get_fc1_out_features(self):
        """获取全连接层输出维度 / Get FC layer output features."""
        return self.config.getint('MODEL_ARCHITECTURE', 'fc1_out_features')

    def get_dropout_conv(self):
        """获取卷积层 Dropout 率 / Get convolutional Dropout rate."""
        return self.config.getfloat('MODEL_ARCHITECTURE', 'dropout_conv', fallback=0.0)

    def get_dropout_fc(self):
        """获取全连接层 Dropout 率 / Get FC Dropout rate."""
        return self.config.getfloat('MODEL_ARCHITECTURE', 'dropout_fc', fallback=0.0)

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
        """获取数据集文件路径 / Get dataset file path."""
        filename = self.config.get('DATA_GENERATION', 'output_filename')
        return os.path.join(self.get_output_path(), filename)

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
        default_str = self.config.get('TRAINING_PARAMETER_RANGES', 'default_range', fallback='-3.0, 3.0')
        default_min, default_max = map(float, default_str.split(','))

        ranges = {}
        for name in self.get_trainable_param_names():
            range_str = self.config.get('TRAINING_PARAMETER_RANGES', name, fallback=None)
            if range_str:
                try:
                    min_val, max_val = map(float, range_str.split(','))
                    ranges[name] = (min_val, max_val)
                except ValueError:
                    ranges[name] = (default_min, default_max)
            else:
                ranges[name] = (default_min, default_max)
        return ranges

    # ==================== NANOROBOT_MODELING 纳米机器人建模 ====================

    def get_experimental_data_path(self):
        """获取实验数据路径 / Get experimental data path."""
        path = self.config.get('NANOROBOT_MODELING', 'path_to_experimental_data_a')
        if not os.path.isabs(path):
            if path.startswith('./'):
                path = path[2:]
            if os.path.dirname(path):
                return self._to_cwd_relative(self._resolve_from_config_dir(path))
            return os.path.join(self.get_output_path(), path)
        return self._to_cwd_relative(path)

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

    # ==================== AUTOENCODER 自编码器训练参数 ====================

    def get_decoder_num_epochs(self):
        """获取 Decoder 训练轮数 / Get Decoder training epochs."""
        return self.config.getint('AUTOENCODER', 'decoder_num_epochs', fallback=500)

    def get_decoder_lr(self):
        """获取 Decoder 学习率 / Get Decoder learning rate."""
        return self.config.getfloat('AUTOENCODER', 'decoder_lr', fallback=0.001)

    def get_decoder_patience(self):
        """获取 Decoder Early Stopping 耐心值 / Get Decoder early stopping patience."""
        return self.config.getint('AUTOENCODER', 'decoder_patience', fallback=100)

    def get_alpha_recon(self):
        """获取重构 Loss 权重 α / Get reconstruction loss weight α."""
        return self.config.getfloat('AUTOENCODER', 'alpha_recon', fallback=0.7)

    def get_beta_param(self):
        """获取参数 Loss 权重 β / Get parameter loss weight β."""
        return self.config.getfloat('AUTOENCODER', 'beta_param', fallback=0.3)


if __name__ == "__main__":
    # 配置加载测试 / Config loading test
    try:
        config = Config()
        print("=== 配置加载测试 / Config Loading Test ===")
        print(f"Input size:  {config.get_input_size()}")
        print(f"Output size: {config.get_output_size()}")
        print(f"LR:          {config.get_learning_rate()}")
        print(f"Batch size:  {config.get_batch_size()}")
        print(f"Params:      {config.get_trainable_param_names()}")
        print(f"Conv1:       {config.get_conv1_params()}")
        print(f"SG window:   {config.get_sg_window()}")
        print(f"Ensemble:    {config.get_ensemble_size()}")
        print(f"Decoder LR:  {config.get_decoder_lr()}")
        print("\n配置加载成功 / Config loaded successfully!")
    except Exception as e:
        print(f"配置加载失败 / Config load failed: {e}")
