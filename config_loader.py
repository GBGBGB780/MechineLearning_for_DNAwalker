# coding=utf-8
import configparser
import os


class Config:
    """统一的配置管理类"""
    
    def __init__(self, config_file='configfile.ini'):
        """
        初始化配置加载器
        
        Args:
            config_file: 配置文件路径，默认为 'configfile.ini'
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件 {config_file} 不存在")
        
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8')
        self.config_file = config_file
    
    # ==================== TRAINING 参数 ====================
    
    def get_sim_duration_minutes(self):
        """获取模拟持续时间（分钟）"""
        return self.config.getint('TRAINING', 'sim_duration_minutes')
    
    def get_num_curves(self):
        """获取输入曲线数量"""
        return self.config.getint('TRAINING', 'num_curves')
    
    def get_seq_length(self):
        """
        获取时间序列长度
        自动计算: sim_duration_minutes * 60 + 1
        例如: 130分钟 * 60秒/分钟 + 1个初始点 = 7801
        """
        duration_minutes = self.get_sim_duration_minutes()
        return duration_minutes * 60 + 1
    
    def get_input_size(self):
        """
        获取输入维度 (自动计算: num_curves * seq_length)
        例如: 3条曲线 * 7801个时间点 = 23403
        """
        return self.get_num_curves() * self.get_seq_length()
    
    def get_output_size(self):
        """获取输出维度 (待预测的物理参数数量)"""
        return self.config.getint('TRAINING', 'output_size')
    
    def get_learning_rate(self):
        """获取初始学习率"""
        return self.config.getfloat('TRAINING', 'learning_rate')
    
    def get_batch_size(self):
        """获取批次大小"""
        return self.config.getint('TRAINING', 'batch_size')
    
    def get_num_epochs(self):
        """获取训练轮数"""
        return self.config.getint('TRAINING', 'num_epochs')
    
    def get_model_save_path(self):
        """获取模型保存路径"""
        return self.config.get('TRAINING', 'model_save_path')
    
    def get_x_scaler_file(self):
        """获取X数据归一化器保存路径"""
        return self.config.get('TRAINING', 'x_scaler_file')
    
    def get_y_scaler_file(self):
        """获取Y数据归一化器保存路径"""
        return self.config.get('TRAINING', 'y_scaler_file')
    
    # 学习率调度器参数
    def get_scheduler_mode(self):
        """获取学习率调度器模式 (min/max)"""
        return self.config.get('TRAINING', 'scheduler_mode')
    
    def get_scheduler_factor(self):
        """获取学习率衰减因子"""
        return self.config.getfloat('TRAINING', 'scheduler_factor')
    
    def get_scheduler_patience(self):
        """获取学习率调度器耐心值 (多少轮不降就调整)"""
        return self.config.getint('TRAINING', 'scheduler_patience')
    
    def get_scheduler_min_lr(self):
        """获取最小学习率"""
        return self.config.getfloat('TRAINING', 'scheduler_min_lr')
    
    # 数据集拆分参数
    def get_test_split_ratio(self):
        """获取测试集比例"""
        return self.config.getfloat('TRAINING', 'test_split_ratio')
    
    def get_val_split_ratio(self):
        """获取验证集比例"""
        return self.config.getfloat('TRAINING', 'val_split_ratio')
    
    def get_random_seed(self):
        """获取随机种子"""
        return self.config.getint('TRAINING', 'random_seed')
    
    # ==================== MODEL_ARCHITECTURE 参数 ====================
    # 注意: num_curves 和 seq_length 现在从 TRAINING 节读取
    
    # 卷积层参数
    def get_conv1_params(self):
        """获取第一层卷积参数"""
        return {
            'out_channels': self.config.getint('MODEL_ARCHITECTURE', 'conv1_out_channels'),
            'kernel_size': self.config.getint('MODEL_ARCHITECTURE', 'conv1_kernel_size'),
            'stride': self.config.getint('MODEL_ARCHITECTURE', 'conv1_stride'),
            'padding': self.config.getint('MODEL_ARCHITECTURE', 'conv1_padding')
        }
    
    def get_conv2_params(self):
        """获取第二层卷积参数"""
        return {
            'out_channels': self.config.getint('MODEL_ARCHITECTURE', 'conv2_out_channels'),
            'kernel_size': self.config.getint('MODEL_ARCHITECTURE', 'conv2_kernel_size'),
            'stride': self.config.getint('MODEL_ARCHITECTURE', 'conv2_stride'),
            'padding': self.config.getint('MODEL_ARCHITECTURE', 'conv2_padding')
        }
    
    def get_conv3_params(self):
        """获取第三层卷积参数"""
        return {
            'out_channels': self.config.getint('MODEL_ARCHITECTURE', 'conv3_out_channels'),
            'kernel_size': self.config.getint('MODEL_ARCHITECTURE', 'conv3_kernel_size'),
            'stride': self.config.getint('MODEL_ARCHITECTURE', 'conv3_stride'),
            'padding': self.config.getint('MODEL_ARCHITECTURE', 'conv3_padding')
        }
    
    def get_conv4_params(self):
        """获取第四层卷积参数"""
        return {
            'out_channels': self.config.getint('MODEL_ARCHITECTURE', 'conv4_out_channels'),
            'kernel_size': self.config.getint('MODEL_ARCHITECTURE', 'conv4_kernel_size'),
            'stride': self.config.getint('MODEL_ARCHITECTURE', 'conv4_stride'),
            'padding': self.config.getint('MODEL_ARCHITECTURE', 'conv4_padding')
        }
    
    def get_fc1_out_features(self):
        """获取第一层全连接层输出维度"""
        return self.config.getint('MODEL_ARCHITECTURE', 'fc1_out_features')
    
    # ==================== DATA_PROCESSING 参数 ====================
    
    def get_safe_threshold(self):
        """获取数据清洗安全阈值"""
        return self.config.getfloat('DATA_PROCESSING', 'safe_threshold')
    
    def get_nan_replacement_value(self):
        """获取NaN替换值"""
        return self.config.getfloat('DATA_PROCESSING', 'nan_replacement_value')
    
    def get_log_epsilon(self):
        """获取log变换时的极小值"""
        return self.config.getfloat('DATA_PROCESSING', 'log_epsilon')
    
    # ==================== DATA_GENERATION 参数 ====================
    
    def get_num_time_points(self):
        """获取时间点数量"""
        return self.config.getint('DATA_GENERATION', 'num_time_points')
    
    def get_dataset_file(self):
        """获取数据集文件名"""
        return self.config.get('DATA_GENERATION', 'output_filename')
    
    # ==================== PHYSICAL_PARAMETERS ====================
    
    def get_trainable_param_names(self):
        """
        从PHYSICAL_PARAMETERS中获取需要训练的参数名称
        (值为空的参数即为待训练参数)
        
        Returns:
            list: 待训练参数名称列表
        """
        params = []
        for key in self.config['PHYSICAL_PARAMETERS']:
            value = self.config['PHYSICAL_PARAMETERS'][key].strip()
            # 空值或只有注释的参数视为需要训练
            if value == '' or value.startswith('#'):
                params.append(key)
        return params
    
    def get_p_unbind_track(self):
        """获取轨道解绑概率"""
        return self.config.getfloat('PHYSICAL_PARAMETERS', 'p_unbind_track')
    
    # ==================== NANOROBOT_MODELING ====================
    
    def get_experimental_data_path(self):
        """获取实验数据路径"""
        return self.config.get('NANOROBOT_MODELING', 'path_to_experimental_data_a')
    
    def get_sim_total_time(self):
        """获取模拟总时长"""
        return self.config.getfloat('NANOROBOT_MODELING', 'sim_total_time')


if __name__ == "__main__":
    # 测试配置加载
    try:
        config = Config()
        print("=== 配置加载测试 ===")
        print(f"输入维度: {config.get_input_size()}")
        print(f"输出维度: {config.get_output_size()}")
        print(f"学习率: {config.get_learning_rate()}")
        print(f"批次大小: {config.get_batch_size()}")
        print(f"训练轮数: {config.get_num_epochs()}")
        print(f"待训练参数: {config.get_trainable_param_names()}")
        print(f"第一层卷积参数: {config.get_conv1_params()}")
        print("\n配置加载成功!")
    except Exception as e:
        print(f"配置加载失败: {e}")
