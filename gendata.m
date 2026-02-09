% -------------------------------------------------------------------------
% 脚本：generate_dna_motor_dataset.m
% 目的：使用 parfor 并行生成 DNA 纳米机器人模拟的训练数据集。
%
% 描述：
% 1. 为 7 个物理参数定义范围。
% 2. 使用 LHS 生成 10,000 个参数样本 (Y)。
% 3. 并行运行 10,000 次模拟，每次使用一组参数。
% 4. 模拟逻辑基于 "mechanics_and_kinetics_of_the_winding_DNA_motor...m"。
% 5. 将每个模拟的3条荧光曲线插值到 100 个标准时间点 (X)。
% 6. 将 X, Y 和 param_names 保存到 'training_dataset.mat'。
% -------------------------------------------------------------------------

%% 1. 设置和配置
clc;
clear;
rng('default'); % 确保 LHS 采样的可复现性

disp('开始生成数据集...');
tic; % 开始计时

% --- 从您的需求中定义 ---
target_num_samples = 15000;      % 目标合格样本总数
initial_sample_ratio = 2.0;       % 初始采样冗余比例（生成2倍样本以应对质量过滤）
num_samples = round(target_num_samples * initial_sample_ratio);  % 初始生成样本数

simu_time = 130;    % 模拟时间为130min
num_time_points = simu_time * 60 + 1;     % 标准化的时间点数量，总时间130min，因此7800点（秒），加上t=0，所以7801
output_filename = 'training_dataset.mat';
if exist(output_filename, 'file')
    delete(output_filename); % 删除旧文件，确保从头开始
end

fprintf('=== 数据集生成配置 ===\n');
fprintf('目标合格样本数: %d\n', target_num_samples);
fprintf('初始生成样本数: %d (冗余比例 %.1f)\n', num_samples, initial_sample_ratio);
fprintf('====================\n\n');

%% 2. 待训练的参数定义 (来自 configfile.ini) 
% 7个待训练参数的名称
param_names = {
    'E_b',
    'E_b_azo_trans',
    'E_b_azo_cis',
    'k_mig',
    'k0',
    'drt_z',
    'drt_s'
};

% 对应的最小值范围
min_vals = [
    -2.0,
    -2.0,
    -1.0,
    0.01,
    1e-6,
    0.001,
    0.001
];

% 对应的最大值范围
max_vals = [
    -0.5,
    -0.5,
    -0.001,
    1.0,
    1e-4,
    1.0,
    1.0
];
min_vals = min_vals(:).';
max_vals = max_vals(:).';

%% 3. 拉丁超立方采样 (LHS)
disp(['正在生成 ', num2str(num_samples), ' 组 LHS 参数样本...']);

% 生成 [0, 1] 范围内的归一化样本
lhs_design_normalized = lhsdesign(num_samples, length(param_names));

% Y: (10000, 7) 输入参数矩阵
% 将样本缩放到正确的物理范围
Y = min_vals + (max_vals - min_vals) .* lhs_design_normalized;

%% 4. 定义固定参数 (来自 configfile.ini 和原始脚本)
% 将所有不改变的参数打包到一个结构体中，以便传入 parfor
disp('正在设置固定的模拟参数...');

% 基础物理性质参数
fixed_params.kBT = 4.14;
fixed_params.lp_s = 0.75;
fixed_params.lc_s = 0.7;
fixed_params.lc_d = 0.34;
fixed_params.di_DNA = 2;
fixed_params.dE_TYE = -1.55;
fixed_params.p_unbind_track = 0.09507;

% 结构设计参数
fixed_params.n_D1 = 10;
fixed_params.n_D2 = 10;
fixed_params.n_S1 = 4;
fixed_params.n_gray = 10;
fixed_params.n_hairpin_1 = 8;
fixed_params.n_hairpin_2 = 8;
fixed_params.n_azo_1 = 3;
fixed_params.n_azo_2 = 3;
fixed_params.n_T_hairpin_1 = 3;
fixed_params.n_T_hairpin_2 = 2;
fixed_params.n_track_1 = 15;
fixed_params.n_track_2 = 55;

%% 5. 并行模拟
disp('正在启动并行计算池...');
if isempty(gcp('nocreate'))
    parpool; % 如果没有活动的池，则启动一个
end
disp('并行池已启动。');

disp(['正在并行运行 ', num2str(num_samples), ' 次模拟...']);

% X: (10000, 3, 100) 输出曲线矩阵
% (样本, 曲线索引, 时间点)
% 索引 1: FAM, 2: TYE, 3: CY5
X = zeros(num_samples, 3, num_time_points);

% 为了在 parfor 中高效传递，创建 fixed_params 的副本
fp = fixed_params;

X_local = cell(num_samples, 1);

% Setup progress tracking
D = parallel.pool.DataQueue;
afterEach(D, @(~) updateProgress(num_samples, false));
updateProgress(num_samples, true); % Initialize

parfor i = 1:num_samples
    % 1. 准备参数
    current_params = struct(...
        'E_b',           Y(i, 1), ...
        'E_b_azo_trans', Y(i, 2), ...
        'E_b_azo_cis',   Y(i, 3), ...
        'k_mig',         Y(i, 4), ...
        'k0',            Y(i, 5), ...
        'drt_z',         Y(i, 6), ...
        'drt_s',         Y(i, 7) ...
    );

    % 2. 模拟
    [time, fam, tye, cy5, dt_used] = run_dna_motor_simulation(current_params, fp);
    % 将结果放入 cell 中（每个 cell 是一个 struct，包含信号和 dt）
    X_local{i} = struct('signals', [fam, tye, cy5]', 'dt_used', dt_used);

    send(D, []);
end

% 4. 提取数据并合并
% 分离信号和 dt 信息
dt_records = zeros(num_samples, 1);
X_temp = cell(num_samples, 1);

for i = 1:num_samples
    X_temp{i} = X_local{i}.signals;
    dt_records(i) = X_local{i}.dt_used;
end

% 合并 cell 为三维矩阵
X = cat(3, X_temp{:});
X = permute(X, [3,1,2]); % 调整维度，使其成为 (num_samples, 3, num_time_points)

% 立即释放内存
clear X_local X_temp;

disp('...所有并行模拟已完成。');

%% 5. 数据质量验证和补充
disp('========================================');
disp('正在进行数据质量验证...');

% 5.1 验证初始样本 (分批处理以节省内存)
valid_indices = true(num_samples, 1);
invalid_reasons = cell(num_samples, 1);

MIN_DT_THRESHOLD = 1.2e-5;  % dt 阈值，与模拟函数中的 MIN_DT 一致

batch_size = 5000; % 为了防止内存溢出，分批进行检测
fprintf('开始分批质量检测 (批大小: %d)...\n', batch_size);

for b_start = 1:batch_size:num_samples
    b_end = min(b_start + batch_size - 1, num_samples);
    current_idx = b_start:b_end;
    current_count = length(current_idx);
    
    % 提取当前批次数据 (避免一次性 broadcasting 整个 X)
    X_batch = X(current_idx, :, :);
    dt_batch = dt_records(current_idx);
    
    valid_batch = true(current_count, 1);
    reasons_batch = cell(current_count, 1);
    
    % 并行处理当前批次
    parfor k = 1:current_count
        % 调用验证函数
        % 调用验证函数
        % [修正] 正确切片: (k, 通道, :) -> squeeze 得到 (时间点, 1)
        [valid_batch(k), reasons_batch{k}] = validate_single_sample(...
            squeeze(X_batch(k, 1, :)), squeeze(X_batch(k, 2, :)), squeeze(X_batch(k, 3, :)), ...
            dt_batch(k), MIN_DT_THRESHOLD);
    end
    
    valid_indices(current_idx) = valid_batch;
    invalid_reasons(current_idx) = reasons_batch;
    
    fprintf('  批次 %d-%d 完成检测。\n', b_start, b_end);
end

num_valid = sum(valid_indices);
num_invalid = num_samples - num_valid;

fprintf('\n--- 初始样本质量统计 ---\n');
fprintf('合格样本数: %d / %d (%.1f%%)\n', num_valid, num_samples, 100*num_valid/num_samples);
fprintf('不合格样本数: %d (%.1f%%)\n', num_invalid, 100*num_invalid/num_samples);

% 统计不合格原因
if num_invalid > 0
    fprintf('\n不合格原因分布:\n');
    unique_reasons = unique(invalid_reasons(~cellfun(@isempty, invalid_reasons)));
    for k = 1:length(unique_reasons)
        count = sum(strcmp(invalid_reasons, unique_reasons{k}));
        fprintf('  - %s: %d 个样本 (%.1f%%)\n', unique_reasons{k}, count, 100*count/num_invalid);
    end
end

% 5.2 立即保存合格样本并释放内存
fprintf('正在将合格样本写入文件以释放内存...\n');

% 1. 初始化 matfile (如果尚未存在)
if ~exist(output_filename, 'file')
    save(output_filename, 'param_names', '-v7.3'); % 先保存 param_names
end
m = matfile(output_filename, 'Writable', true);

% 检查当前文件里已经存了多少 (如果重新运行可能需要追加，或者这里每次都覆盖? 
% 假设每次从头运行，这里初始化计数)
% 为了安全，我们在开头 delete 了文件或者 clear 了，这里假设从 0 开始
total_saved = 0; % 记录已保存到磁盘的合格样本数

if num_valid > 0
    X_valid = X(valid_indices, :, :);
    Y_valid = Y(valid_indices, :);
    
    [n_new, ~, ~] = size(X_valid);
    
    % 写入 matfile (X_final, Y_final)
    % 注意: matfile 支持部分写入
    m.X_final(1:n_new, :, :) = X_valid;
    m.Y_final(1:n_new, :) = Y_valid;
    
    total_saved = n_new;
    fprintf('已保存初始批次 %d 个合格样本。\n', n_new);
end

% 记录目前为止生成的样本总数(含无效)，用于增量LHS
total_generated_count = num_samples;

% 彻底清理内存中的大数组
clear X Y dt_records valid_indices invalid_reasons X_valid Y_valid;

% 更新 num_valid 为已保存的数量
num_valid = total_saved;

% 5.3 补充样本逻辑
round_num = 1;
max_rounds = 500;  % 最大补充轮数，防止无限循环，设置一个比较大的数保证足够合格数据

while num_valid < target_num_samples && round_num <= max_rounds
    needed = target_num_samples - num_valid;
    extra_generate = ceil(needed * 1.6);  % 多生成60%作为冗余

    fprintf('\n========================================\n');
    fprintf('=== 第 %d 轮补充采样 ===\n', round_num);
    fprintf('当前合格数: %d / %d\n', num_valid, target_num_samples);
    fprintf('需要补充: %d 个样本\n', needed);
    fprintf('实际生成: %d 个样本 (含60%%冗余)\n', extra_generate);
    fprintf('========================================\n');

    % 生成新的 LHS 样本 (策略 1: 增量式 LHS / 分层排除法)
    % 原理: 生成 (Total_Old + New) 个样本，取最后 New 个，以保持整体 LHS 空间特性
    current_total_needed = total_generated_count + extra_generate;
    
    disp(['正在生成增量 LHS 样本 (', num2str(total_generated_count), ' -> ', num2str(current_total_needed), ')...']);
    
    % 生成更大的 LHS 样本集
    lhs_full = lhsdesign(current_total_needed, length(param_names));
    
    % 截取新增的部分
    lhs_additional = lhs_full(total_generated_count + 1 : end, :);
    
    % 更新总生成计数
    total_generated_count = current_total_needed;
    
    Y_additional = min_vals + (max_vals - min_vals) .* lhs_additional;

    % 运行补充模拟
    X_additional_local = cell(extra_generate, 1);

    % Setup supplemental progress tracking
    D_supp = parallel.pool.DataQueue;
    afterEach(D_supp, @(~) updateProgress(extra_generate, false));
    updateProgress(extra_generate, true); % Initialize
    
    parfor i = 1:extra_generate
        current_params = struct(...
            'E_b',           Y_additional(i, 1), ...
            'E_b_azo_trans', Y_additional(i, 2), ...
            'E_b_azo_cis',   Y_additional(i, 3), ...
            'k_mig',         Y_additional(i, 4), ...
            'k0',            Y_additional(i, 5), ...
            'drt_z',         Y_additional(i, 6), ...
            'drt_s',         Y_additional(i, 7) ...
        );

        [time, fam, tye, cy5, dt_used] = run_dna_motor_simulation(current_params, fp);
        X_additional_local{i} = struct('signals', [fam, tye, cy5]', 'dt_used', dt_used);

        send(D_supp, []);
    end

    % 合并并验证补充样本
    % 分离信号和 dt 信息
    dt_additional = zeros(extra_generate, 1);
    X_additional_temp = cell(extra_generate, 1);

    for i = 1:extra_generate
        X_additional_temp{i} = X_additional_local{i}.signals;
        dt_additional(i) = X_additional_local{i}.dt_used;
    end

    X_additional = cat(3, X_additional_temp{:});
    X_additional = permute(X_additional, [3,1,2]);
    
    % 立即释放内存
    clear X_additional_local X_additional_temp;

    valid_additional = true(extra_generate, 1);
    invalid_additional = cell(extra_generate, 1);

    parfor i = 1:extra_generate
        fam = squeeze(X_additional(i, 1, :));
        tye = squeeze(X_additional(i, 2, :));
        cy5 = squeeze(X_additional(i, 3, :));
        dt_this = dt_additional(i);

        % --- 使用封装的验证函数 ---
        [valid_additional(i), invalid_additional{i}] = validate_single_sample(...
            fam, tye, cy5, dt_this, MIN_DT_THRESHOLD);
    end

    num_valid_additional = sum(valid_additional);
    fprintf('补充样本合格数: %d / %d (%.1f%%)\n', ...
        num_valid_additional, extra_generate, 100*num_valid_additional/extra_generate);

    % 仅合并合格的补充样本
    if num_valid_additional > 0
        X_additional_valid = X_additional(valid_additional, :, :);
        Y_additional_valid = Y_additional(valid_additional, :);
        
        % 写入 matfile
        start_idx = total_saved + 1;
        end_idx = total_saved + num_valid_additional;
        
        m.X_final(start_idx:end_idx, :, :) = X_additional_valid;
        m.Y_final(start_idx:end_idx, :) = Y_additional_valid;
        
        total_saved = total_saved + num_valid_additional;
        num_valid = total_saved;
    end
    
    % 清理临时变量
    clear X_additional Y_additional dt_additional valid_additional invalid_additional X_additional_valid Y_additional_valid lhs_full lhs_additional;

    fprintf('累计合格样本数: %d / %d\n', num_valid, target_num_samples);

    round_num = round_num + 1;
end

% 5.3 最终筛选和保存
fprintf('\n========================================\n');
if num_valid >= target_num_samples
    fprintf('✓ 成功获得足够的合格样本！\n');
    fprintf('最终样本数: %d (文件中已保存)\n', num_valid);
    
    % 如果需要精确等于 target_num_samples，可以在这里截断文件
    % 但 matfile 不支持 shrink，通常多一点没关系，或者在此处提示
    if num_valid > target_num_samples
        fprintf('提示: 实际样本数 (%d) 略多于目标数 (%d)，这通常是可以接受的。\n', num_valid, target_num_samples);
    end
else
    fprintf('⚠ 警告：未能获得足够的合格样本！\n');
    fprintf('目标: %d, 实际获得: %d\n', target_num_samples, num_valid);
    fprintf('已保存所有合格样本。\n');
end
fprintf('========================================\n\n');

% 关闭并行池
% delete(gcp('nocreate'));
% disp('并行池已关闭。');

%% 6. 结束
toc; % 结束计时
disp('----------------------------------------------------');
disp('数据集生成完毕！');
disp(['文件已保存为: ', output_filename]);
disp('包含变量: X_final, Y_final, param_names');
disp('注意: 数据已使用 matfile 增量写入，无需再次 save。');

toc; % 结束计时
disp('----------------------------------------------------');
disp('数据集生成完毕！');
disp(['文件已保存为: ', output_filename]);
disp('包含变量:');
fprintf('  X_final (输出): [%d, %d, %d] (样本, 曲线, 时间点)\n', size(X_final));
fprintf('  Y_final (输入): [%d, %d] (样本, 参数)\n', size(Y_final));
disp('  param_names (标签): [1, 7] (参数名称)');
fprintf('\n数据质量保证:\n');
fprintf('  ✓ 所有样本均无 NaN/Inf 值\n');
fprintf('  ✓ FAM 曲线变化 ≥ 0.02\n');
fprintf('  ✓ TYE 曲线变化 ≥ 0.6\n');
fprintf('  ✓ CY5 曲线变化 ≥ 0.02\n');
disp('----------------------------------------------------');
disp('模拟完成, 即将退出。');
exit;


%% 8. 单样本验证函数 (本地函数)
function [is_valid, reason] = validate_single_sample(fam, tye, cy5, dt_this, min_dt_threshold)
    % 统一的样本质量验证逻辑
    % 输入: 3条曲线, 实际dt, 最小dt阈值
    % 输出: 是否合格, 拒绝原因

    is_valid = true;
    reason = '';

    % 检查 0: dt 是否被强制调整
    if dt_this <= min_dt_threshold
        is_valid = false;
        reason = 'dt too small (slow dynamics)';
    end

    % 检查 1: NaN/Inf
    if is_valid && (any(isnan(fam)) || any(isnan(tye)) || any(isnan(cy5)) || ...
       any(isinf(fam)) || any(isinf(tye)) || any(isinf(cy5)))
        is_valid = false;
        reason = 'Contains NaN or Inf';
    end

    % 检查 2: 曲线变化范围
    if is_valid
        fam_change = max(fam) - min(fam);
        tye_change = max(tye) - min(tye);
        cy5_change = max(cy5) - min(cy5);

        % 使用 <= 判断是否"变化过小" (根据用户要求)
        if fam_change <= 0.02
            is_valid = false;
            reason = 'FAM change <= 0.02';
        elseif tye_change <= 0.6
            is_valid = false;
            reason = 'TYE change <= 0.6';
        elseif cy5_change <= 0.02
            is_valid = false;
            reason = 'CY5 change <= 0.02';
        end
    end
end


%% 7. 核心模拟函数 (本地函数)
function [time_out, fam_out, tye_out, cy5_out, dt_used] = run_dna_motor_simulation(sim_params, fixed_params)
    % 该函数在 parfor 循环的每个 worker 上独立运行
    % 返回值 dt_used: 实际使用的时间步长，用于质量检测

    % -------------------------------------------------
    % 1. 解包参数
    % -------------------------------------------------

    % 从 sim_params (可变) 中解包
    E_b = sim_params.E_b;
    E_b_azo_trans = sim_params.E_b_azo_trans;
    E_b_azo_cis = sim_params.E_b_azo_cis;
    k_mig = sim_params.k_mig;
    k0 = sim_params.k0;
    drt_z = sim_params.drt_z;
    drt_s = sim_params.drt_s;

    % 从 fixed_params (固定) 中解包
    kBT = fixed_params.kBT;
    lp_s = fixed_params.lp_s;
    lc_s = fixed_params.lc_s;
    lc_d = fixed_params.lc_d;
    di_DNA = fixed_params.di_DNA;
    dE_TYE = fixed_params.dE_TYE;
    p_unbind_track = fixed_params.p_unbind_track;
    n_D1 = fixed_params.n_D1;
    n_D2 = fixed_params.n_D2;
    n_S1 = fixed_params.n_S1;
    n_gray = fixed_params.n_gray;
    n_hairpin_1 = fixed_params.n_hairpin_1;
    n_hairpin_2 = fixed_params.n_hairpin_2;
    n_azo_1 = fixed_params.n_azo_1;
    n_azo_2 = fixed_params.n_azo_2;
    n_T_hairpin_1 = fixed_params.n_T_hairpin_1;
    n_T_hairpin_2 = fixed_params.n_T_hairpin_2;
    n_track_1 = fixed_params.n_track_1;
    n_track_2 = fixed_params.n_track_2;

    % -------------------------------------------------
    % 2. 原始脚本的物理和动力学计算
    % (从 "mechanics_and_kinetics_..." 文件中粘贴)
    % -------------------------------------------------

    %**************************************************************************
    %                    configuration free energy

    %..........................................................................
    %            binding configuraton and energy of the shearing foot
    E_shear_foot=100;
    for i=0:1:n_D2
    n_D2_detach=i;
    E_b_shear=E_b*(n_D1+n_D2-n_D2_detach);
    %n_track_1*lc_d-di_DNA
    x=(n_track_1*lc_d)/(lc_s*(2*n_D2_detach+n_D1));
        if x<1
            E_shear=E_b_shear+(lc_s*(2*n_D2_detach+n_D1))*x^2*(3-2*x)/4/(1-x);
        else
            E_shear=1000;
        end
        if E_shear_foot>E_shear
            E_shear_foot=E_shear;
            n_shear_foot=n_D1+n_D2-n_D2_detach;
        end
    end
    %E_shear_foot
    %n_shear_foot
    E_zipper_foot=E_b*(n_D1+n_D2);
    %..........................................................................
    %                     single foot binding

    E_config_t(1)=E_zipper_foot;
    E_config_t(2)=E_shear_foot;
    E_config_c(1)=E_zipper_foot;
    E_config_c(2)=E_shear_foot;

    %..........................................................................
    %                      double feet binding

    %..........................................................................
    %state 3:

    E_state_min_t=1000;
    f_state_min_t=0.0;
    E_state_min_c=1000;
    f_state_min_c=0.0;
    for i=1:1:n_hairpin_1+n_hairpin_2

        n_hairpin_open=i;

        if n_hairpin_open<n_hairpin_1
            x=((n_track_1+n_track_2-2*n_gray)*lc_d)/((n_hairpin_open)*2*lc_s);
            n_chain=n_hairpin_open;
        elseif (n_hairpin_open>=n_hairpin_1 && n_hairpin_open<n_hairpin_1+n_hairpin_2)
            x=((n_track_1+n_track_2-2*n_gray)*lc_d)/((n_hairpin_open+n_T_hairpin_1)*2*lc_s);
            n_chain=n_hairpin_open++n_T_hairpin_1;
        else
            x=((n_track_1+n_track_2-2*n_gray)*lc_d)/((n_hairpin_open+n_T_hairpin_1+n_T_hairpin_2)*2*lc_s);
            n_chain=n_hairpin_open++n_T_hairpin_1+n_T_hairpin_2;
        end

        if x<1
            E_neck=2*((n_chain*2*lc_s)/(lp_s))*x^2*(3-2*x)/4/(1-x);
            f_state=2*kBT/lp_s*(x-0.25+(1-x)^-2/4);
        else
            E_neck=1000;
            f_state=1000;
        end
        E_state_t=E_neck+2*E_zipper_foot-2*n_hairpin_open*E_b_azo_trans;
        E_state_c=E_neck+2*E_zipper_foot-2*n_hairpin_open*E_b_azo_cis;

        if E_state_min_t>E_state_t
            E_state_min_t=E_state_t;
            f_state_min_t=f_state;
            n_state_open_t=i;
        end
        if E_state_min_c>E_state_c
            E_state_min_c=E_state_c;
            f_state_min_c=f_state;
            n_state_open_c=i;
        end
    end
    E_config_t(3)=E_state_min_t;
    f_config_t(3)=f_state_min_t;
    n_config_t(3)=n_state_open_t;
    E_config_c(3)=E_state_min_c;
    f_config_c(3)=f_state_min_c;
    n_config_c(3)=n_state_open_c;

    %..........................................................................
    %state 4:

    E_state_min_t=1000;
    f_state_min_t=0.0;
    E_state_min_c=1000;
    f_state_min_c=0.0;
    for i=1:1:n_hairpin_1+n_hairpin_2

        n_hairpin_open=i;

        if n_hairpin_open<n_hairpin_1
            x=((n_track_1+n_track_2-2*n_gray)*lc_d)/((n_hairpin_open)*2*lc_s);
            n_chain=n_hairpin_open;
        elseif (n_hairpin_open>=n_hairpin_1 && n_hairpin_open<n_hairpin_1+n_hairpin_2)
            x=((n_track_1+n_track_2-2*n_gray)*lc_d)/((n_hairpin_open+n_T_hairpin_1)*2*lc_s);
            n_chain=n_hairpin_open++n_T_hairpin_1;
        else
            x=((n_track_1+n_track_2-2*n_gray)*lc_d)/((n_hairpin_open+n_T_hairpin_1+n_T_hairpin_2)*2*lc_s);
            n_chain=n_hairpin_open++n_T_hairpin_1+n_T_hairpin_2;
        end

        if x<1
            E_neck=2*((n_chain*2*lc_s)/(lp_s))*x^2*(3-2*x)/4/(1-x);
            f_state=2*kBT/lp_s*(x-0.25+(1-x)^-2/4);
        else
            E_neck=1000;
            f_state=1000;
        end
        E_state_t=E_neck+2*E_shear_foot-2*n_hairpin_open*E_b_azo_trans;
        E_state_c=E_neck+2*E_shear_foot-2*n_hairpin_open*E_b_azo_cis;

        if E_state_min_t>E_state_t
            E_state_min_t=E_state_t;
            f_state_min_t=f_state;
            n_state_open_t=i;
        end
        if E_state_min_c>E_state_c
            E_state_min_c=E_state_c;
            f_state_min_c=f_state;
            n_state_open_c=i;
        end
    end
    E_config_t(4)=E_state_min_t;
    f_config_t(4)=f_state_min_t;
    n_config_t(4)=n_state_open_t;
    E_config_c(4)=E_state_min_c;
    f_config_c(4)=f_state_min_c;
    n_config_c(4)=n_state_open_c;

    %..........................................................................
    %state 5:

    E_state_min_t=1000;
    f_state_min_t=0.0;
    E_state_min_c=1000;
    f_state_min_c=0.0;
    for i=1:1:n_hairpin_1+n_hairpin_2

        n_hairpin_open=i;

        if n_hairpin_open<n_hairpin_1
            x=((n_track_2-2*n_gray)*lc_d)/((n_hairpin_open)*2*lc_s);
            n_chain=n_hairpin_open;
        elseif (n_hairpin_open>=n_hairpin_1 && n_hairpin_open<n_hairpin_1+n_hairpin_2)
            x=((n_track_2-2*n_gray)*lc_d)/((n_hairpin_open+n_T_hairpin_1)*2*lc_s);
            n_chain=n_hairpin_open++n_T_hairpin_1;
        else
            x=((n_track_2-2*n_gray)*lc_d)/((n_hairpin_open+n_T_hairpin_1+n_T_hairpin_2)*2*lc_s);
            n_chain=n_hairpin_open++n_T_hairpin_1+n_T_hairpin_2;
        end

        if x<1
            E_neck=2*((n_chain*2*lc_s)/(lp_s))*x^2*(3-2*x)/4/(1-x);
            f_state=2*kBT/lp_s*(x-0.25+(1-x)^-2/4);
        else
            E_neck=1000;
            f_state=1000;
        end
        E_state_t=E_neck+E_zipper_foot+E_shear_foot-2*n_hairpin_open*E_b_azo_trans;
        E_state_c=E_neck+E_zipper_foot+E_shear_foot-2*n_hairpin_open*E_b_azo_cis;

        if E_state_min_t>E_state_t
            E_state_min_t=E_state_t;
            f_state_min_t=f_state;
            n_state_open_t=i;
        end
        if E_state_min_c>E_state_c
            E_state_min_c=E_state_c;
            f_state_min_c=f_state;
            n_state_open_c=i;
        end

    end
    E_config_t(5)=E_state_min_t;
    f_config_t(5)=f_state_min_t;
    n_config_t(5)=n_state_open_t;
    E_config_c(5)=E_state_min_c;
    f_config_c(5)=f_state_min_c;
    n_config_c(5)=n_state_open_c;

    %..........................................................................
    %state 6:

    E_state_min_t=1000;
    f_state_min_t=0.0;
    E_state_min_c=1000;
    f_state_min_c=0.0;
    for i=1:1:n_hairpin_1+n_hairpin_2

        n_hairpin_open=i;

        if n_hairpin_open<n_hairpin_1
            x=((2*n_track_1+n_track_2-2*n_gray)*lc_d)/((n_hairpin_open)*2*lc_s);
            n_chain=n_hairpin_open;
        elseif (n_hairpin_open>=n_hairpin_1 && n_hairpin_open<n_hairpin_1+n_hairpin_2)
            x=((2*n_track_1+n_track_2-2*n_gray)*lc_d)/((n_hairpin_open+n_T_hairpin_1)*2*lc_s);
            n_chain=n_hairpin_open++n_T_hairpin_1;
        else
            x=((2*n_track_1+n_track_2-2*n_gray)*lc_d)/((n_hairpin_open+n_T_hairpin_1+n_T_hairpin_2)*2*lc_s);
            n_chain=n_hairpin_open++n_T_hairpin_1+n_T_hairpin_2;
        end

        if x<1
            E_neck=2*((n_chain*2*lc_s)/(lp_s))*x^2*(3-2*x)/4/(1-x);
            f_state=2*kBT/lp_s*(x-0.25+(1-x)^-2/4);
        else
            E_neck=1000;
            f_state=1000;
        end
        E_state_t=E_neck+E_zipper_foot+E_shear_foot-2*n_hairpin_open*E_b_azo_trans;
        E_state_c=E_neck+E_zipper_foot+E_shear_foot-2*n_hairpin_open*E_b_azo_cis;

        if E_state_min_t>E_state_t
            E_state_min_t=E_state_t;
            f_state_min_t=f_state;
            n_state_open_t=i;
        end
        if E_state_min_c>E_state_c
            E_state_min_c=E_state_c;
            f_state_min_c=f_state;
            n_state_open_c=i;
        end
    end
    E_config_t(6)=E_state_min_t;
    f_config_t(6)=f_state_min_t;
    n_config_t(6)=n_state_open_t;
    E_config_c(6)=E_state_min_c;
    f_config_c(6)=f_state_min_c;
    n_config_c(6)=n_state_open_c;



    %**************************************************************************
    %                              kinetics

    %..........................................................................
    %        energy mapping from 6 states to 14 states on 3 binding-site track

    E_config_t_copy=E_config_t;
    E_config_c_copy=E_config_c;
    f_config_t_copy=f_config_t;
    f_config_c_copy=f_config_c;

    E_config_t(1:3)=E_config_t_copy(1);   E_config_t(4:6)=E_config_t_copy(2);
    E_config_t(7:8)=E_config_t_copy(3);   E_config_t(9:10)=E_config_t_copy(4);
    E_config_t(11:12)=E_config_t_copy(5); E_config_t(13:14)=E_config_t_copy(6);

    E_config_c(1:3)=E_config_c_copy(1);   E_config_c(4:6)=E_config_c_copy(2);
    E_config_c(7:8)=E_config_c_copy(3);   E_config_c(9:10)=E_config_c_copy(4);
    E_config_c(11:12)=E_config_c_copy(5); E_config_c(13:14)=E_config_c_copy(6);

    f_config_t(1:3)=f_config_t_copy(1);   f_config_t(4:6)=f_config_t_copy(2);
    f_config_t(7:8)=f_config_t_copy(3);   f_config_t(9:10)=f_config_t_copy(4);
    f_config_t(11:12)=f_config_t_copy(5); f_config_t(13:14)=f_config_t_copy(6);

    f_config_c(1:3)=f_config_c_copy(1);   f_config_c(4:6)=f_config_c_copy(2);
    f_config_c(7:8)=f_config_c_copy(3);   f_config_c(9:10)=f_config_c_copy(4);
    f_config_c(11:12)=f_config_c_copy(5); f_config_c(13:14)=f_config_c_copy(6);

    E_config_t(1)=E_config_t(1)+dE_TYE;
    E_config_t(4)=E_config_t(4)+dE_TYE;
    E_config_t(7)=E_config_t(7)+dE_TYE;
    E_config_t(9)=E_config_t(9)+dE_TYE;
    E_config_t(11)=E_config_t(11)+dE_TYE;
    E_config_t(13)=E_config_t(13)+dE_TYE;

    E_config_c(1)=E_config_c(1)+dE_TYE;
    E_config_c(4)=E_config_c(4)+dE_TYE;
    E_config_c(7)=E_config_c(7)+dE_TYE;
    E_config_c(9)=E_config_c(9)+dE_TYE;
    E_config_c(11)=E_config_c(11)+dE_TYE;
    E_config_c(13)=E_config_c(13)+dE_TYE;

    %..........................................................................
    %                 transition rate (k) matrix

    k_trans = zeros(14, 14);
    k_cis = zeros(14, 14);

    %trans:
    %1,2 means 1 to 2

    %single-single
    k_trans(4,1)=k_mig;     k_trans(5,2)=k_mig;     k_trans(6,3)=k_mig;
    k_trans(1,4)=k_trans(4,1)*exp(E_config_t(1)-E_config_t(4));
    k_trans(2,5)=k_trans(5,2)*exp(E_config_t(2)-E_config_t(5));
    k_trans(3,6)=k_trans(6,3)*exp(E_config_t(3)-E_config_t(6));

    %single-double
    k_trans(7,1)=k0*exp(f_config_t(7)*drt_z/kBT);  k_trans(11,1)=k0*exp(f_config_t(11)*drt_s/kBT);
    k_trans(1,7)=k_trans(7,1)*exp(E_config_t(1)-E_config_t(7));
    k_trans(1,11)=k_trans(11,1)*exp(E_config_t(1)-E_config_t(11));

    k_trans(7,2)=k0*exp(f_config_t(7)*drt_z/kBT);  k_trans(8,2)=k0*exp(f_config_t(8)*drt_z/kBT);
    k_trans(12,2)=k0*exp(f_config_t(12)*drt_s/kBT);  k_trans(13,2)=k0*exp(f_config_t(13)*drt_s/kBT);
    k_trans(2,7)=k_trans(7,2)*exp(E_config_t(2)-E_config_t(7));
    k_trans(2,8)=k_trans(8,2)*exp(E_config_t(2)-E_config_t(8));
    k_trans(2,12)=k_trans(12,2)*exp(E_config_t(2)-E_config_t(12));
    k_trans(2,13)=k_trans(13,2)*exp(E_config_t(2)-E_config_t(13));

    k_trans(8,3)=k0*exp(f_config_t(8)*drt_z/kBT);  k_trans(14,3)=k0*exp(f_config_t(14)*drt_s/kBT);
    k_trans(3,8)=k_trans(8,3)*exp(E_config_t(3)-E_config_t(8));
    k_trans(3,14)=k_trans(14,3)*exp(E_config_t(3)-E_config_t(14));

    k_trans(9,4)=k0*exp(f_config_t(9)*drt_s/kBT);  k_trans(13,4)=k0*exp(f_config_t(13)*drt_z/kBT);
    k_trans(4,9)=k_trans(9,4)*exp(E_config_t(4)-E_config_t(9));
    k_trans(4,13)=k_trans(13,4)*exp(E_config_t(4)-E_config_t(13));

    k_trans(9,5)=k0*exp(f_config_t(9)*drt_s/kBT);  k_trans(10,5)=k0*exp(f_config_t(10)*drt_s/kBT);
    k_trans(11,5)=k0*exp(f_config_t(11)*drt_z/kBT);  k_trans(14,5)=k0*exp(f_config_t(14)*drt_z/kBT);
    k_trans(5,9)=k_trans(9,5)*exp(E_config_t(5)-E_config_t(9));
    k_trans(5,10)=k_trans(10,5)*exp(E_config_t(5)-E_config_t(10));
    k_trans(5,11)=k_trans(11,5)*exp(E_config_t(5)-E_config_t(11));
    k_trans(5,14)=k_trans(14,5)*exp(E_config_t(5)-E_config_t(14));

    k_trans(10,6)=k0*exp(f_config_t(10)*drt_s/kBT);  k_trans(12,6)=k0*exp(f_config_t(12)*drt_z/kBT);
    k_trans(6,10)=k_trans(10,6)*exp(E_config_t(6)-E_config_t(10));
    k_trans(6,12)=k_trans(12,6)*exp(E_config_t(6)-E_config_t(12));


    %double-double
    k_trans(7,11)=k_mig;   k_trans(13,7)=k_mig;
    k_trans(11,7)=k_trans(7,11)*exp(E_config_t(11)-E_config_t(7));
    k_trans(7,13)=k_trans(13,7)*exp(E_config_t(7)-E_config_t(13));

    k_trans(8,12)=k_mig;   k_trans(14,8)=k_mig;
    k_trans(12,8)=k_trans(8,12)*exp(E_config_t(12)-E_config_t(8));
    k_trans(8,14)=k_trans(14,8)*exp(E_config_t(8)-E_config_t(14));

    k_trans(9,11)=k_mig;   k_trans(13,9)=k_mig;
    k_trans(11,9)=k_trans(9,11)*exp(E_config_t(11)-E_config_t(9));
    k_trans(9,13)=k_trans(13,9)*exp(E_config_t(9)-E_config_t(13));

    k_trans(10,12)=k_mig;   k_trans(14,10)=k_mig;
    k_trans(12,10)=k_trans(10,12)*exp(E_config_t(12)-E_config_t(10));
    k_trans(10,14)=k_trans(14,10)*exp(E_config_t(10)-E_config_t(14));



    %cis:

    %single-single
    k_cis(4,1)=k_mig;     k_cis(5,2)=k_mig;     k_cis(6,3)=k_mig;
    k_cis(1,4)=k_cis(4,1)*exp(E_config_c(1)-E_config_c(4));
    k_cis(2,5)=k_cis(5,2)*exp(E_config_c(2)-E_config_c(5));
    k_cis(3,6)=k_cis(6,3)*exp(E_config_c(3)-E_config_c(6));

    %single-double
    k_cis(7,1)=k0*exp(f_config_c(7)*drt_z/kBT);  k_cis(11,1)=k0*exp(f_config_c(11)*drt_s/kBT);
    k_cis(1,7)=k_cis(7,1)*exp(E_config_c(1)-E_config_c(7));
    k_cis(1,11)=k_cis(11,1)*exp(E_config_c(1)-E_config_c(11));

    k_cis(7,2)=k0*exp(f_config_c(7)*drt_z/kBT);  k_cis(8,2)=k0*exp(f_config_c(8)*drt_z/kBT);
    k_cis(12,2)=k0*exp(f_config_c(12)*drt_s/kBT);  k_cis(13,2)=k0*exp(f_config_c(13)*drt_s/kBT);
    k_cis(2,7)=k_cis(7,2)*exp(E_config_c(2)-E_config_c(7));
    k_cis(2,8)=k_cis(8,2)*exp(E_config_c(2)-E_config_c(8));
    k_cis(2,12)=k_cis(12,2)*exp(E_config_c(2)-E_config_c(12));
    k_cis(2,13)=k_cis(13,2)*exp(E_config_c(2)-E_config_c(13));

    k_cis(8,3)=k0*exp(f_config_c(8)*drt_z/kBT);  k_cis(14,3)=k0*exp(f_config_c(14)*drt_s/kBT);
    k_cis(3,8)=k_cis(8,3)*exp(E_config_c(3)-E_config_c(8));
    k_cis(3,14)=k_cis(14,3)*exp(E_config_c(3)-E_config_c(14));

    k_cis(9,4)=k0*exp(f_config_c(9)*drt_s/kBT);  k_cis(13,4)=k0*exp(f_config_c(13)*drt_z/kBT);
    k_cis(4,9)=k_cis(9,4)*exp(E_config_c(4)-E_config_c(9));
    k_cis(4,13)=k_cis(13,4)*exp(E_config_c(4)-E_config_c(13));

    k_cis(9,5)=k0*exp(f_config_c(9)*drt_s/kBT);  k_cis(10,5)=k0*exp(f_config_c(10)*drt_s/kBT);
    k_cis(11,5)=k0*exp(f_config_c(11)*drt_z/kBT);  k_cis(14,5)=k0*exp(f_config_c(14)*drt_z/kBT);
    k_cis(5,9)=k_cis(9,5)*exp(E_config_c(5)-E_config_c(9));
    k_cis(5,10)=k_cis(10,5)*exp(E_config_c(5)-E_config_c(10));
    k_cis(5,11)=k_cis(11,5)*exp(E_config_c(5)-E_config_c(11));
    k_cis(5,14)=k_cis(14,5)*exp(E_config_c(5)-E_config_c(14));

    k_cis(10,6)=k0*exp(f_config_c(10)*drt_s/kBT);  k_cis(12,6)=k0*exp(f_config_c(12)*drt_z/kBT);
    k_cis(6,10)=k_cis(10,6)*exp(E_config_c(6)-E_config_c(10));
    k_cis(6,12)=k_cis(12,6)*exp(E_config_c(6)-E_config_c(12));


    %double-double
    k_cis(7,11)=k_mig;   k_cis(13,7)=k_mig;
    k_cis(11,7)=k_cis(7,11)*exp(E_config_c(11)-E_config_c(7));
    k_cis(7,13)=k_cis(13,7)*exp(E_config_c(7)-E_config_c(13));

    k_cis(8,12)=k_mig;   k_cis(14,8)=k_mig;
    k_cis(12,8)=k_cis(8,12)*exp(E_config_c(12)-E_config_c(8));
    k_cis(8,14)=k_cis(14,8)*exp(E_config_c(8)-E_config_c(14));

    k_cis(9,11)=k_mig;   k_cis(13,9)=k_mig;
    k_cis(11,9)=k_cis(9,11)*exp(E_config_c(11)-E_config_c(9));
    k_cis(9,13)=k_cis(13,9)*exp(E_config_c(9)-E_config_c(13));

    k_cis(10,12)=k_mig;   k_cis(14,10)=k_mig;
    k_cis(12,10)=k_cis(10,12)*exp(E_config_c(12)-E_config_c(10));
    k_cis(10,14)=k_cis(14,10)*exp(E_config_c(10)-E_config_c(14));

    %..........................................................................
    %                     master equation R matrix
    mag_t=floor(log10(max(max(max(k_cis)),max(max(k_trans)))));
    dt=1/10^mag_t/10;

    % --- [防护代码开始] ---
    % 设定一个最小 dt (例如 1.2e-6)，
    % 对应最大循环步数 (200*60)/1.2e-6 = 1e10
    MIN_DT = 1.2e-5; %此处如果运行速度较慢可以将阈值降低比如1.2e-3
    %1.2e-5实测24核cpu需要12小时左右

    if dt < MIN_DT
        dt = MIN_DT;
    end

    if isnan(dt) || isinf(dt) || dt == 0
        dt = 1e-4; % 备用安全值，如果上面修改了这里也需要进行修改比如1e-2
    end

    % 记录使用的 dt（用于返回）
    dt_used = dt;
    % --- [防护代码结束] ---

    R_vis = zeros(14, 14);
    R_UV = zeros(14, 14);

    for i=1:1:14
        for j=1:1:14
            if i==j
                R_vis(i,i)=1;
                R_UV(i,i)=1;
                for nn=1:1:14
                     R_vis(i,i)=R_vis(i,i)-k_trans(i,nn)*dt;
                     R_UV(i,i)=R_UV(i,i)-k_cis(i,nn)*dt;
                end
            else
                R_vis(i,j)=k_trans(j,i)*dt;
                R_UV(i,j)=k_cis(j,i)*dt;
            end
        end
    end
    %..........................................................................
    %                           initial condition
    p_total=0.945;

    p_config=zeros(1,14);

    p_config(11:12)=exp(-E_config_c(11:12)-20);
    pp=sum(p_config);
    if pp == 0
        % 避免除以零，如果 E_config_c 太大
        p_config(11:12) = p_total / 2; % 均匀分配
    else
        p_config(11:12)=p_config(11:12)/pp*p_total;
    end

    p_config=p_config';

    %..........................................................................
    %                               dynamics

    % 预分配结果矩阵以提高效率
    simu_time = 130;
    save_interval_min = 1/60;
    num_results = simu_time / save_interval_min + 1;
    result_signal = zeros(num_results, 4);

    i_result=1;

    total_steps = round(simu_time * 60 / dt);

    % 为循环外的变量初始化
    dp14=0;
    dp25=0;
    dp36=0;
    dp511=0;
    dp212=0;
    b_forward=0;
    b_backward=0;

    for i=0:1:total_steps

        current_time_min = i * dt / 60;

        % 确定是 Vis 还是 UV
        if mod(floor(current_time_min / 10)+1, 2) == 1
            R_dyn=R_vis;
            k_dyn=k_trans;
        else
            R_dyn=R_UV;
            k_dyn=k_cis;
        end

        if i~=0
            p_config_old=p_config;
            p_config=R_dyn*p_config;
        end


        % 保存逻辑：每 60 秒（1 分钟）保存一次
        % 使用一个小的容差来处理浮点数精度问题
        if mod(current_time_min, save_interval_min) < (dt / 60)

            if i_result > num_results
                % 超出了预期的数据点，可能是由于浮点数问题
                break;
            end

            signal_FAM=p_config(1)+p_config(2)+p_config(4)+p_config(5)+p_config(7)+p_config(11)+p_config(9)+p_config(13);
            signal_TYE=p_config(2)+p_config(3)+p_config(5)+p_config(6)+p_config(8)+p_config(12)+p_config(10)+p_config(14);
            signal_CY5=p_config(1)+p_config(3)+p_config(4)+p_config(6)+p_unbind_track;

            result_signal(i_result, 1)=current_time_min;
            result_signal(i_result, 2)=signal_FAM;
            result_signal(i_result, 3)=signal_TYE;
            result_signal(i_result, 4)=signal_CY5;

            i_result=i_result+1;
        end

    end % 结束动力学循环

    % -------------------------------------------------
    % 3. 准备函数输出
    % -------------------------------------------------
    % 原始脚本会保存许多 .txt 文件，在这里我们跳过这些
    % 我们只返回生成训练集所需的核心数据
    time_out = result_signal(:, 1);
    fam_out = result_signal(:, 2);
    tye_out = result_signal(:, 3);
    cy5_out = result_signal(:, 4);
end % 结束 run_dna_motor_simulation 函数

%% 9. 进度条更新函数 (本地函数)
function updateProgress(total_samples, is_init)
    persistent p
    
    if nargin > 1 && is_init
        p = 0;
        fprintf('进度: 0.0%%\n');
        return;
    end
    
    if isempty(p)
        p = 0;
    end
    p = p + 1;
    
    % 每 1% 更新一次
    if mod(p, ceil(total_samples/100)) == 0 || p == total_samples
        fprintf('进度: %.1f%%\n', p/total_samples*100);
    end
end