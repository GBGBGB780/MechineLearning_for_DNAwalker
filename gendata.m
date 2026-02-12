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
% 优化：采用增量分批生成策略，防止内存溢出 (OOM)。
% 包含两个阶段：
% 1. 初始采样：分批生成初始设定数量的样本。
% 2. 补充采样：根据有效样本数量，分批生成补充样本，直至达到目标。
% -------------------------------------------------------------------------

%% 1. 设置和配置
clc;
clear;
rng('default'); % 确保 LHS 采样的可复现性

disp('开始生成数据集...');
tic; % 开始计时

% --- 从您的需求中定义 ---
target_num_samples = 15000;      % 目标合格样本总数
initial_sample_ratio = 3.0;       % 初始采样冗余比例（生成3倍样本以应对质量过滤）
num_samples = round(target_num_samples * initial_sample_ratio);  % 初始生成样本数

simu_time = 130;    % 模拟时间为130min
num_time_points = simu_time * 60 + 1;     % 标准化的时间点数量，总时间130min，因此7800点（秒），加上t=0，所以7801
output_filename = 'training_dataset.mat';
if exist(output_filename, 'file')
    delete(output_filename); % 删除旧文件，确保从头开始
end

% --- 内存优化配置 ---
MAX_BATCH_SIZE = 10000; % 每批次最大样本数，防止内存溢出

fprintf('=== 数据集生成配置 ===\n');
fprintf('目标合格样本数: %d\n', target_num_samples);
fprintf('初始生成样本数: %d (冗余比例 %.1f)\n', num_samples, initial_sample_ratio);
fprintf('批次大小限制: %d\n', MAX_BATCH_SIZE);
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

%% 3. 定义固定参数 (来自 configfile.ini 和原始脚本)
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

%% 4. 初始化
disp('正在启动并行计算池...');
if isempty(gcp('nocreate'))
    parpool; % 如果没有活动的池，则启动一个
end
disp('并行池已启动。');

% 初始化 matfile
if ~exist(output_filename, 'file')
    save(output_filename, 'param_names', '-v7.3'); % 先保存 param_names
end
m = matfile(output_filename, 'Writable', true);

total_saved = 0; % 记录已保存到磁盘的合格样本数
total_generated_count = 0; % 记录已生成的总样本数(含无效)，用于LHS索引

%% 5. 第一阶段：初始采样 (分批处理)
disp('=== 阶段 1: 初始采样 ===');
disp(['计划生成总数: ', num2str(num_samples)]);

% 生成第一阶段所有的 参数 (Y)
lhs_design_normalized = lhsdesign(num_samples, length(param_names));
Y_all_initial = min_vals + (max_vals - min_vals) .* lhs_design_normalized;

% 将初始任务分批
num_batches_init = ceil(num_samples / MAX_BATCH_SIZE);

for b = 1:num_batches_init
    % 计算当前批次的索引范围
    start_idx = (b-1) * MAX_BATCH_SIZE + 1;
    end_idx = min(b * MAX_BATCH_SIZE, num_samples);
    current_batch_size = end_idx - start_idx + 1;
    
    fprintf('\n--- 初始采样: 正在处理第 %d / %d 批 (样本 %d - %d, 数量: %d) ---\n', ...
        b, num_batches_init, start_idx, end_idx, current_batch_size);
    
    % 提取当前批次的 Y
    Y_batch = Y_all_initial(start_idx:end_idx, :);
    
    % 运行模拟、验证并保存
    [n_ok, ~] = run_batch_simulation_and_save(Y_batch, fixed_params, m, ...
        num_time_points, 1.2e-5, sprintf('初始批次 %d', b));
    
    total_saved = total_saved + n_ok;
    total_generated_count = total_generated_count + current_batch_size;
end

disp(['初始阶段完成。当前合格样本数: ', num2str(total_saved)]);

%% 6. 第二阶段：补充采样 (分批处理)
round_num = 1;
max_rounds = 500;  % 最大补充轮数，防止无限循环，设置一个比较大的数保证足够合格数据

while total_saved < target_num_samples && round_num <= max_rounds
    needed = target_num_samples - total_saved;
    extra_generate = ceil(needed * 4);  % 生成400%作为冗余

    fprintf('\n========================================\n');
    fprintf('=== 第 %d 轮补充采样 ===\n', round_num);
    fprintf('当前合格数: %d / %d\n', total_saved, target_num_samples);
    fprintf('需要补充: %d 个样本\n', needed);
    fprintf('计划生成: %d 个样本 (含400%%冗余)\n', extra_generate);
    fprintf('========================================\n');

    % 1. 生成增量 LHS 参数 (一次性生成本轮所需的 Y)
    current_total_needed_LHS = total_generated_count + extra_generate;
    disp(['正在生成增量 LHS 参数 (', num2str(total_generated_count), ' -> ', num2str(current_total_needed_LHS), ')...']);
    
    lhs_full = lhsdesign(current_total_needed_LHS, length(param_names));
    lhs_additional = lhs_full(total_generated_count + 1 : end, :);
    Y_additional_all = min_vals + (max_vals - min_vals) .* lhs_additional;
    
    clear lhs_full lhs_additional; % 及时清理
    
    % 2. 将本轮的补充任务分批执行
    num_batches_supp = ceil(extra_generate / MAX_BATCH_SIZE);
    
    for b = 1:num_batches_supp
        start_idx = (b-1) * MAX_BATCH_SIZE + 1;
        end_idx = min(b * MAX_BATCH_SIZE, extra_generate);
        current_batch_size = end_idx - start_idx + 1;
        
        fprintf('\n--- 补充采样(轮次 %d): 正在处理第 %d / %d 批 (数量: %d) ---\n', ...
            round_num, b, num_batches_supp, current_batch_size);
        
        % 提取当前批次的 Y
        Y_batch = Y_additional_all(start_idx:end_idx, :);
        
        % 运行模拟、验证并保存
        [n_ok, ~] = run_batch_simulation_and_save(Y_batch, fixed_params, m, ...
            num_time_points, 1.2e-5, sprintf('补充(轮%d)批次%d', round_num, b));
        
        total_saved = total_saved + n_ok;
        total_generated_count = total_generated_count + current_batch_size;
        
        % 如果已经满足目标，提前退出批次循环
        if total_saved >= target_num_samples
            fprintf('>>> 此批次已达成目标样本数，停止本轮剩余批次。\n');
            break;
        end
    end
    
    round_num = round_num + 1;
end

%% 7. 结束
toc;
disp('----------------------------------------------------');
if total_saved >= target_num_samples
    fprintf('✓ 成功获得足够的合格样本！\n');
else
    fprintf('⚠ 警告：未能获得足够的合格样本！\n');
end
fprintf('最终样本数: %d (文件中已保存)\n', total_saved);
disp(['文件已保存为: ', output_filename]);
disp('----------------------------------------------------');
try 
    x_sz = size(m.X_final);
    y_sz = size(m.Y_final);
    fprintf('  X_final (输出): [%d, %d, %d] (样本, 曲线, 时间点)\n', x_sz);
    fprintf('  Y_final (输入): [%d, %d] (样本, 参数)\n', y_sz);
catch
    fprintf('  (无法读取 X_final/Y_final 大小，可能未生成任何合格样本)\n');
end
disp('  param_names (标签): [1, 7] (参数名称)');
fprintf('\n数据质量保证:\n');
fprintf('  ✓ 所有样本均无 NaN/Inf 值\n');
fprintf('  ✓ FAM 曲线变化 ≥ 0.02\n');
fprintf('  ✓ TYE 曲线变化 ≥ 0.6\n');
fprintf('  ✓ CY5 曲线变化 ≥ 0.02\n');
disp('----------------------------------------------------');
disp('模拟完成, 即将退出。');
exit;


%% 8. 核心批处理流程函数 (新增)
function [num_valid_saved, valid_indices_local] = run_batch_simulation_and_save(Y_batch, fixed_params, mfile_obj, num_time_points, min_dt_threshold, batch_label)
    % 1. 准备
    n_samples = size(Y_batch, 1);
    X_local_cell = cell(n_samples, 1);
    
    % 为了 parfor
    fp = fixed_params; 
    
    % 2. 模拟 (Parfor)
    parfor i = 1:n_samples
        current_params = struct(...
            'E_b',           Y_batch(i, 1), ...
            'E_b_azo_trans', Y_batch(i, 2), ...
            'E_b_azo_cis',   Y_batch(i, 3), ...
            'k_mig',         Y_batch(i, 4), ...
            'k0',            Y_batch(i, 5), ...
            'drt_z',         Y_batch(i, 6), ...
            'drt_s',         Y_batch(i, 7) ...
        );
        [~, fam, tye, cy5, dt_used] = run_dna_motor_simulation(current_params, fp);
        X_local_cell{i} = struct('signals', [fam, tye, cy5]', 'dt_used', dt_used);
    end
    
    % 3. 整理结果
    dt_records = zeros(n_samples, 1);
    X_temp_merge = cell(n_samples, 1);
    for i = 1:n_samples
        X_temp_merge{i} = X_local_cell{i}.signals;
        dt_records(i) = X_local_cell{i}.dt_used;
    end
    X_batch = cat(3, X_temp_merge{:});
    X_batch = permute(X_batch, [3,1,2]); % (N, 3, T)
    
    clear X_local_cell X_temp_merge; % 释放
    
    % 4. 验证
    [valid_indices_local, invalid_reasons, num_valid] = validate_and_report_stats(...
        X_batch, dt_records, n_samples, min_dt_threshold, batch_label);
    
    num_valid_saved = 0;
    
    % 5. 保存
    if num_valid > 0
        X_to_save = X_batch(valid_indices_local, :, :);
        Y_to_save = Y_batch(valid_indices_local, :);
        [n_new, ~, ~] = size(X_to_save);
        
        % 获取当前文件中的记录数，以确定追加位置
        try
            [dim1, ~] = size(mfile_obj, 'Y_final');
            current_file_count = dim1;
        catch
            current_file_count = 0;
        end
        
        start_idx = current_file_count + 1;
        end_idx = current_file_count + n_new;
        
        % 第一次写入时的初始化
        if current_file_count == 0
             mfile_obj.X_final = X_to_save;
             mfile_obj.Y_final = Y_to_save;
        else
             mfile_obj.X_final(start_idx:end_idx, :, :) = X_to_save;
             mfile_obj.Y_final(start_idx:end_idx, :) = Y_to_save;
        end
        
        num_valid_saved = n_new;
        fprintf('  -> 已追加保存 %d 个合格样本 (当前文件总数: %d)\n', n_new, end_idx);
    else
        fprintf('  -> 本批次无合格样本，未保存。\n');
    end
end

%% 9. 批量验证函数 (本地函数 - 保持不变)
function [valid_indices, invalid_reasons, num_valid] = validate_and_report_stats(X_data, dt_data, batch_size, min_dt_threshold, label_str)
    num_samples = size(X_data, 1);
    valid_indices = true(num_samples, 1);
    invalid_reasons = cell(num_samples, 1);
    
    fprintf('\n--- %s 质量检测 (总数: %d) ---\n', label_str, num_samples);
    
    valid_batch = true(num_samples, 1);
    reasons_batch = cell(num_samples, 1);
        
    parfor k = 1:num_samples
        [valid_batch(k), reasons_batch{k}] = validate_single_sample(...
            squeeze(X_data(k, 1, :)), squeeze(X_data(k, 2, :)), squeeze(X_data(k, 3, :)), ...
            dt_data(k), min_dt_threshold);
    end
        
    valid_indices = valid_batch;
    invalid_reasons = reasons_batch;
    
    num_valid = sum(valid_indices);
    num_invalid = num_samples - num_valid;
    
    fprintf('%s 统计结果: 合格 %d, 不合格 %d\n', label_str, num_valid, num_invalid);
    
    if num_invalid > 0
        fprintf('  不合格原因分布:\n');
        unique_reasons = unique(invalid_reasons(~cellfun(@isempty, invalid_reasons)));
        for k = 1:length(unique_reasons)
            count = sum(strcmp(invalid_reasons, unique_reasons{k}));
            fprintf('    - %s: %d (%.1f%%)\n', unique_reasons{k}, count, 100*count/num_invalid);
        end
    end
end

%% 10. 单样本验证函数 (本地函数 - 保持不变)
function [is_valid, reason] = validate_single_sample(fam, tye, cy5, dt_this, min_dt_threshold)
    is_valid = true;
    reason = '';

    if dt_this <= min_dt_threshold
        is_valid = false;
        reason = 'dt too small';
    end

    if is_valid && (any(isnan(fam)) || any(isnan(tye)) || any(isnan(cy5)) || ...
       any(isinf(fam)) || any(isinf(tye)) || any(isinf(cy5)))
        is_valid = false;
        reason = 'Contains NaN or Inf';
    end

    if is_valid
        fam_change = max(fam) - min(fam);
        tye_change = max(tye) - min(tye);
        cy5_change = max(cy5) - min(cy5);

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

%% 11. 核心模拟函数 (本地函数 - 保持不变)
function [time_out, fam_out, tye_out, cy5_out, dt_used] = run_dna_motor_simulation(sim_params, fixed_params)
    % 解包参数
    E_b = sim_params.E_b;
    E_b_azo_trans = sim_params.E_b_azo_trans;
    E_b_azo_cis = sim_params.E_b_azo_cis;
    k_mig = sim_params.k_mig;
    k0 = sim_params.k0;
    drt_z = sim_params.drt_z;
    drt_s = sim_params.drt_s;

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

    % ... (物理计算部分，长代码，保持原样) ...
    
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

    E_state_min_t=1000; f_state_min_t=0.0; E_state_min_c=1000; f_state_min_c=0.0;
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
            E_neck=1000; f_state=1000;
        end
        E_state_t=E_neck+2*E_zipper_foot-2*n_hairpin_open*E_b_azo_trans;
        E_state_c=E_neck+2*E_zipper_foot-2*n_hairpin_open*E_b_azo_cis;
        if E_state_min_t>E_state_t
            E_state_min_t=E_state_t; f_state_min_t=f_state; n_state_open_t=i;
        end
        if E_state_min_c>E_state_c
            E_state_min_c=E_state_c; f_state_min_c=f_state; n_state_open_c=i;
        end
    end
    E_config_t(3)=E_state_min_t; f_config_t(3)=f_state_min_t; n_config_t(3)=n_state_open_t;
    E_config_c(3)=E_state_min_c; f_config_c(3)=f_state_min_c; n_config_c(3)=n_state_open_c;

    %..........................................................................
    %state 4:

    E_state_min_t=1000; f_state_min_t=0.0; E_state_min_c=1000; f_state_min_c=0.0;
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
            E_neck=1000; f_state=1000;
        end
        E_state_t=E_neck+2*E_shear_foot-2*n_hairpin_open*E_b_azo_trans;
        E_state_c=E_neck+2*E_shear_foot-2*n_hairpin_open*E_b_azo_cis;
        if E_state_min_t>E_state_t
            E_state_min_t=E_state_t; f_state_min_t=f_state; n_state_open_t=i;
        end
        if E_state_min_c>E_state_c
            E_state_min_c=E_state_c; f_state_min_c=f_state; n_state_open_c=i;
        end
    end
    E_config_t(4)=E_state_min_t; f_config_t(4)=f_state_min_t; n_config_t(4)=n_state_open_t;
    E_config_c(4)=E_state_min_c; f_config_c(4)=f_state_min_c; n_config_c(4)=n_state_open_c;

    %..........................................................................
    %state 5:

    E_state_min_t=1000; f_state_min_t=0.0; E_state_min_c=1000; f_state_min_c=0.0;
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
            E_neck=1000; f_state=1000;
        end
        E_state_t=E_neck+E_zipper_foot+E_shear_foot-2*n_hairpin_open*E_b_azo_trans;
        E_state_c=E_neck+E_zipper_foot+E_shear_foot-2*n_hairpin_open*E_b_azo_cis;
        if E_state_min_t>E_state_t
            E_state_min_t=E_state_t; f_state_min_t=f_state; n_state_open_t=i;
        end
        if E_state_min_c>E_state_c
            E_state_min_c=E_state_c; f_state_min_c=f_state; n_state_open_c=i;
        end

    end
    E_config_t(5)=E_state_min_t; f_config_t(5)=f_state_min_t; n_config_t(5)=n_state_open_t;
    E_config_c(5)=E_state_min_c; f_config_c(5)=f_state_min_c; n_config_c(5)=n_state_open_c;

    %..........................................................................
    %state 6:

    E_state_min_t=1000; f_state_min_t=0.0; E_state_min_c=1000; f_state_min_c=0.0;
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
            E_neck=1000; f_state=1000;
        end
        E_state_t=E_neck+E_zipper_foot+E_shear_foot-2*n_hairpin_open*E_b_azo_trans;
        E_state_c=E_neck+E_zipper_foot+E_shear_foot-2*n_hairpin_open*E_b_azo_cis;
        if E_state_min_t>E_state_t
            E_state_min_t=E_state_t; f_state_min_t=f_state; n_state_open_t=i;
        end
        if E_state_min_c>E_state_c
            E_state_min_c=E_state_c; f_state_min_c=f_state; n_state_open_c=i;
        end
    end
    E_config_t(6)=E_state_min_t; f_config_t(6)=f_state_min_t; n_config_t(6)=n_state_open_t;
    E_config_c(6)=E_state_min_c; f_config_c(6)=f_state_min_c; n_config_c(6)=n_state_open_c;



    %**************************************************************************
    %                              kinetics

    %..........................................................................
    %        energy mapping from 6 states to 14 states on 3 binding-site track

    E_config_t_copy=E_config_t; E_config_c_copy=E_config_c;
    f_config_t_copy=f_config_t; f_config_c_copy=f_config_c;

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

    E_config_t(1)=E_config_t(1)+dE_TYE; E_config_t(4)=E_config_t(4)+dE_TYE;
    E_config_t(7)=E_config_t(7)+dE_TYE; E_config_t(9)=E_config_t(9)+dE_TYE;
    E_config_t(11)=E_config_t(11)+dE_TYE; E_config_t(13)=E_config_t(13)+dE_TYE;

    E_config_c(1)=E_config_c(1)+dE_TYE; E_config_c(4)=E_config_c(4)+dE_TYE;
    E_config_c(7)=E_config_c(7)+dE_TYE; E_config_c(9)=E_config_c(9)+dE_TYE;
    E_config_c(11)=E_config_c(11)+dE_TYE; E_config_c(13)=E_config_c(13)+dE_TYE;

    %..........................................................................
    %                 transition rate (k) matrix

    k_trans = zeros(14, 14); k_cis = zeros(14, 14);

    %trans:
    %1,2 means 1 to 2

    %single-single
    k_trans(4,1)=k_mig; k_trans(5,2)=k_mig; k_trans(6,3)=k_mig;
    k_trans(1,4)=k_trans(4,1)*exp(E_config_t(1)-E_config_t(4));
    k_trans(2,5)=k_trans(5,2)*exp(E_config_t(2)-E_config_t(5));
    k_trans(3,6)=k_trans(6,3)*exp(E_config_t(3)-E_config_t(6));

    %single-double
    k_trans(7,1)=k0*exp(f_config_t(7)*drt_z/kBT); k_trans(11,1)=k0*exp(f_config_t(11)*drt_s/kBT);
    k_trans(1,7)=k_trans(7,1)*exp(E_config_t(1)-E_config_t(7));
    k_trans(1,11)=k_trans(11,1)*exp(E_config_t(1)-E_config_t(11));

    k_trans(7,2)=k0*exp(f_config_t(7)*drt_z/kBT); k_trans(8,2)=k0*exp(f_config_t(8)*drt_z/kBT);
    k_trans(12,2)=k0*exp(f_config_t(12)*drt_s/kBT); k_trans(13,2)=k0*exp(f_config_t(13)*drt_s/kBT);
    k_trans(2,7)=k_trans(7,2)*exp(E_config_t(2)-E_config_t(7));
    k_trans(2,8)=k_trans(8,2)*exp(E_config_t(2)-E_config_t(8));
    k_trans(2,12)=k_trans(12,2)*exp(E_config_t(2)-E_config_t(12));
    k_trans(2,13)=k_trans(13,2)*exp(E_config_t(2)-E_config_t(13));

    k_trans(8,3)=k0*exp(f_config_t(8)*drt_z/kBT); k_trans(14,3)=k0*exp(f_config_t(14)*drt_s/kBT);
    k_trans(3,8)=k_trans(8,3)*exp(E_config_t(3)-E_config_t(8));
    k_trans(3,14)=k_trans(14,3)*exp(E_config_t(3)-E_config_t(14));

    k_trans(9,4)=k0*exp(f_config_t(9)*drt_s/kBT); k_trans(13,4)=k0*exp(f_config_t(13)*drt_z/kBT);
    k_trans(4,9)=k_trans(9,4)*exp(E_config_t(4)-E_config_t(9));
    k_trans(4,13)=k_trans(13,4)*exp(E_config_t(4)-E_config_t(13));

    k_trans(9,5)=k0*exp(f_config_t(9)*drt_s/kBT); k_trans(10,5)=k0*exp(f_config_t(10)*drt_s/kBT);
    k_trans(11,5)=k0*exp(f_config_t(11)*drt_z/kBT); k_trans(14,5)=k0*exp(f_config_t(14)*drt_z/kBT);
    k_trans(5,9)=k_trans(9,5)*exp(E_config_t(5)-E_config_t(9));
    k_trans(5,10)=k_trans(10,5)*exp(E_config_t(5)-E_config_t(10));
    k_trans(5,11)=k_trans(11,5)*exp(E_config_t(5)-E_config_t(11));
    k_trans(5,14)=k_trans(14,5)*exp(E_config_t(5)-E_config_t(14));

    k_trans(10,6)=k0*exp(f_config_t(10)*drt_s/kBT); k_trans(12,6)=k0*exp(f_config_t(12)*drt_z/kBT);
    k_trans(6,10)=k_trans(10,6)*exp(E_config_t(6)-E_config_t(10));
    k_trans(6,12)=k_trans(12,6)*exp(E_config_t(6)-E_config_t(12));


    %double-double
    k_trans(7,11)=k_mig; k_trans(13,7)=k_mig;
    k_trans(11,7)=k_trans(7,11)*exp(E_config_t(11)-E_config_t(7));
    k_trans(7,13)=k_trans(13,7)*exp(E_config_t(7)-E_config_t(13));

    k_trans(8,12)=k_mig; k_trans(14,8)=k_mig;
    k_trans(12,8)=k_trans(8,12)*exp(E_config_t(12)-E_config_t(8));
    k_trans(8,14)=k_trans(14,8)*exp(E_config_t(8)-E_config_t(14));

    k_trans(9,11)=k_mig; k_trans(13,9)=k_mig;
    k_trans(11,9)=k_trans(9,11)*exp(E_config_t(11)-E_config_t(9));
    k_trans(9,13)=k_trans(13,9)*exp(E_config_t(9)-E_config_t(13));

    k_trans(10,12)=k_mig; k_trans(14,10)=k_mig;
    k_trans(12,10)=k_trans(10,12)*exp(E_config_t(12)-E_config_t(10));
    k_trans(10,14)=k_trans(14,10)*exp(E_config_t(10)-E_config_t(14));


    %cis:

    %single-single
    k_cis(4,1)=k_mig; k_cis(5,2)=k_mig; k_cis(6,3)=k_mig;
    k_cis(1,4)=k_cis(4,1)*exp(E_config_c(1)-E_config_c(4));
    k_cis(2,5)=k_cis(5,2)*exp(E_config_c(2)-E_config_c(5));
    k_cis(3,6)=k_cis(6,3)*exp(E_config_c(3)-E_config_c(6));

    %single-double
    k_cis(7,1)=k0*exp(f_config_c(7)*drt_z/kBT); k_cis(11,1)=k0*exp(f_config_c(11)*drt_s/kBT);
    k_cis(1,7)=k_cis(7,1)*exp(E_config_c(1)-E_config_c(7));
    k_cis(1,11)=k_cis(11,1)*exp(E_config_c(1)-E_config_c(11));

    k_cis(7,2)=k0*exp(f_config_c(7)*drt_z/kBT); k_cis(8,2)=k0*exp(f_config_c(8)*drt_z/kBT);
    k_cis(12,2)=k0*exp(f_config_c(12)*drt_s/kBT); k_cis(13,2)=k0*exp(f_config_c(13)*drt_s/kBT);
    k_cis(2,7)=k_cis(7,2)*exp(E_config_c(2)-E_config_c(7));
    k_cis(2,8)=k_cis(8,2)*exp(E_config_c(2)-E_config_c(8));
    k_cis(2,12)=k_cis(12,2)*exp(E_config_c(2)-E_config_c(12));
    k_cis(2,13)=k_cis(13,2)*exp(E_config_c(2)-E_config_c(13));

    k_cis(8,3)=k0*exp(f_config_c(8)*drt_z/kBT); k_cis(14,3)=k0*exp(f_config_c(14)*drt_s/kBT);
    k_cis(3,8)=k_cis(8,3)*exp(E_config_c(3)-E_config_c(8));
    k_cis(3,14)=k_cis(14,3)*exp(E_config_c(3)-E_config_c(14));

    k_cis(9,4)=k0*exp(f_config_c(9)*drt_s/kBT); k_cis(13,4)=k0*exp(f_config_c(13)*drt_z/kBT);
    k_cis(4,9)=k_cis(9,4)*exp(E_config_c(4)-E_config_c(9));
    k_cis(4,13)=k_cis(13,4)*exp(E_config_c(4)-E_config_c(13));

    k_cis(9,5)=k0*exp(f_config_c(9)*drt_s/kBT); k_cis(10,5)=k0*exp(f_config_c(10)*drt_s/kBT);
    k_cis(11,5)=k0*exp(f_config_c(11)*drt_z/kBT); k_cis(14,5)=k0*exp(f_config_c(14)*drt_z/kBT);
    k_cis(5,9)=k_cis(9,5)*exp(E_config_c(5)-E_config_c(9));
    k_cis(5,10)=k_cis(10,5)*exp(E_config_c(5)-E_config_c(10));
    k_cis(5,11)=k_cis(11,5)*exp(E_config_c(5)-E_config_c(11));
    k_cis(5,14)=k_cis(14,5)*exp(E_config_c(5)-E_config_c(14));

    k_cis(10,6)=k0*exp(f_config_c(10)*drt_s/kBT); k_cis(12,6)=k0*exp(f_config_c(12)*drt_z/kBT);
    k_cis(6,10)=k_cis(10,6)*exp(E_config_c(6)-E_config_c(10));
    k_cis(6,12)=k_cis(12,6)*exp(E_config_c(6)-E_config_c(12));

    %double-double
    k_cis(7,11)=k_mig; k_cis(13,7)=k_mig;
    k_cis(11,7)=k_cis(7,11)*exp(E_config_c(11)-E_config_c(7));
    k_cis(7,13)=k_cis(13,7)*exp(E_config_c(7)-E_config_c(13));

    k_cis(8,12)=k_mig; k_cis(14,8)=k_mig;
    k_cis(12,8)=k_cis(8,12)*exp(E_config_c(12)-E_config_c(8));
    k_cis(8,14)=k_cis(14,8)*exp(E_config_c(8)-E_config_c(14));

    k_cis(9,11)=k_mig; k_cis(13,9)=k_mig;
    k_cis(11,9)=k_cis(9,11)*exp(E_config_c(11)-E_config_c(9));
    k_cis(9,13)=k_cis(13,9)*exp(E_config_c(9)-E_config_c(13));

    k_cis(10,12)=k_mig; k_cis(14,10)=k_mig;
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
                R_vis(i,i)=1; R_UV(i,i)=1;
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
        p_config(11:12) = p_total / 2;
    else
        p_config(11:12)=p_config(11:12)/pp*p_total;
    end
    p_config=p_config';

    %..........................................................................
    %                               dynamics

    simu_time = 130;
    save_interval_min = 1/60;
    num_results = simu_time / save_interval_min + 1;
    result_signal = zeros(num_results, 4);
    i_result=1;
    total_steps = round(simu_time * 60 / dt);

    for i=0:1:total_steps
        current_time_min = i * dt / 60;
        if mod(floor(current_time_min / 10)+1, 2) == 1
            R_dyn=R_vis;
        else
            R_dyn=R_UV;
        end

        if i~=0
            p_config=R_dyn*p_config;
        end

        if mod(current_time_min, save_interval_min) < (dt / 60)
            if i_result > num_results
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
    end

    time_out = result_signal(:, 1);
    fam_out = result_signal(:, 2);
    tye_out = result_signal(:, 3);
    cy5_out = result_signal(:, 4);
end

%% 12. 进度条更新函数 (本地函数 - 保持不变)
function updateProgress(total_samples, is_init)
    persistent p
    if nargin > 1 && is_init
        p = 0; fprintf('进度: 0.0%%\n'); return;
    end
    if isempty(p)
        p = 0;
    end
    p = p + 1;
    if mod(p, ceil(total_samples/100)) == 0 || p == total_samples
        fprintf('进度: %.1f%%\n', p/total_samples*100);
    end
end