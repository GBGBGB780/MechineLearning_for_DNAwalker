clc
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%      optimize_params.m - 物理模型参数精细优化工具
%
%  流程：
%    1. 从 matlab_input_params.txt 读取 ML 预测的初始参数
%    2. 读取实验数据 (Fig3a_fitting.xlsx)
%    3. 用 fminsearch 在物理模型仿真的基础上优化参数
%    4. 输出优化后的参数到 optimized_params.txt
%    5. 绘制优化后的对比图

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ============================================================
%  1. 读取 ML 初始参数
%% ============================================================

input_file = 'matlab_input_params.txt';

if ~exist(input_file, 'file')
    error('错误: 未找到输入文件 %s。请先运行 predict.py 生成参数文件!', input_file);
end

fprintf('--- 1. 从 %s 读取初始参数 ---\n', input_file);
fid = fopen(input_file, 'r');
params = struct();

while ~feof(fid)
    line = fgetl(fid);
    if contains(line, '=')
        parts = strsplit(line, '=');
        param_name = strtrim(parts{1});
        param_value = str2double(strtrim(parts{2}));
        if ~strcmp(param_name, 'END_OF_PARAMS')
            params.(param_name) = param_value;
            fprintf('  %s = %e\n', param_name, param_value);
        end
    end
end
fclose(fid);

% 提取初始参数向量 (fminsearch 用的是向量格式)
% param_vec = [E_b, E_b_azo_trans, E_b_azo_cis, k0, k_mig, drt_z, drt_s]
x0 = [params.e_b, params.e_b_azo_trans, params.e_b_azo_cis, ...
      params.k0,  params.k_mig,         params.drt_z,       params.drt_s];

fprintf('\n初始参数读取完成 (7个参数):\n');
fprintf('  e_b=%.4f  e_b_azo_trans=%.4f  e_b_azo_cis=%.4f\n', x0(1), x0(2), x0(3));
fprintf('  k0=%.4e  k_mig=%.4f  drt_z=%.4f  drt_s=%.4f\n\n', x0(4), x0(5), x0(6), x0(7));

%% ============================================================
%  2. 读取实验数据（计算 MSE 用的目标数据）
%% ============================================================

exp_data_file = 'Fig3a_fitting.xlsx';
if ~exist(exp_data_file, 'file')
    error('错误: 未找到实验数据 %s', exp_data_file);
end
fprintf('--- 2. 读取实验数据 %s ---\n', exp_data_file);

exp_data = readtable(exp_data_file);
exp_time = exp_data.Time;

% 提取各通道实验数据（兼容多种列名格式）
if ismember('FAM_FAMT__', exp_data.Properties.VariableNames)
    exp_fam = exp_data.FAM_FAMT__;
elseif ismember('x_FAM_FAMT__', exp_data.Properties.VariableNames)
    exp_fam = exp_data.x_FAM_FAMT__;
else
    exp_fam = exp_data{:, 2};
end

if ismember('TYE_TYET__', exp_data.Properties.VariableNames)
    exp_tye = exp_data.TYE_TYET__;
elseif ismember('x_TYE_TYET__', exp_data.Properties.VariableNames)
    exp_tye = exp_data.x_TYE_TYET__;
else
    exp_tye = exp_data{:, 3};
end

if ismember('CY5_CY5T_m_', exp_data.Properties.VariableNames)
    exp_cy5 = exp_data.CY5_CY5T_m_;
elseif ismember('x_CY5_CY5T_m_', exp_data.Properties.VariableNames)
    exp_cy5 = exp_data.x_CY5_CY5T_m_;
else
    exp_cy5 = exp_data{:, 4};
end

fprintf('实验数据读取完成: %d 个时间点\n', length(exp_time));

% -----------------------------------------------------------------
% ② 去掉明显的离群点（测量故障 / 尾部漂移点）
% 只保留时间在 [0, 130] 分钟范围内，且各通道信号在物理合理范围内的点
% -----------------------------------------------------------------
valid_mask = exp_time >= 0 & exp_time <= 130 ...
           & exp_fam > 0.5 & exp_fam < 1.0 ...
           & exp_tye > 0.0 & exp_tye < 0.6 ...
           & exp_cy5 > 0.0 & exp_cy5 < 0.4;

n_removed = sum(~valid_mask);
exp_time = exp_time(valid_mask);
exp_fam  = exp_fam(valid_mask);
exp_tye  = exp_tye(valid_mask);
exp_cy5  = exp_cy5(valid_mask);
fprintf('离群点过滤: 移除 %d 个点，保留 %d 个有效时间点\n\n', n_removed, length(exp_time));

%% ============================================================
%  3. 定义物理约束边界（防止优化跑野）
%% ============================================================

% [E_b,   E_b_azo_trans, E_b_azo_cis,  k0,     k_mig, drt_z, drt_s]
lb = [-1.4, -1.2,          -0.18,         2e-6,   0.03,  0.35,  0.02];
ub = [-1.0, -0.8,          -0.03,         2e-5,   0.10,  0.75,  0.10];

%% ============================================================
%  4. 定义目标函数（运行物理仿真并计算 MSE）
%% ============================================================

% 使用计数器跟踪优化进度
eval_count = 0;
best_mse   = inf;

fprintf('--- 3. 开始 fminsearch 物理优化 ---\n');
fprintf('(这可能需要数分钟，每次 MSE 更新时会打印进度)\n\n');

% -----------------------------------------------------------------
% 预计算各通道的线性漂移（polyfit 1阶）
% 优化器对比的是去除漂移后的浮动形状，而不是绝对值
% 这样 FAM/CY5 的盐戒漂移就不会干扰参数估计
% -----------------------------------------------------------------
p_drift_fam = polyfit(exp_time, exp_fam, 1);
p_drift_tye = polyfit(exp_time, exp_tye, 1);
p_drift_cy5 = polyfit(exp_time, exp_cy5, 1);

fprintf('线性漂移预计完成:\n');
fprintf('  FAM 相对倒 = %+.6f/min\n', p_drift_fam(1));
fprintf('  TYE 相对倒 = %+.6f/min\n', p_drift_tye(1));
fprintf('  CY5 相对倒 = %+.6f/min\n\n', p_drift_cy5(1));

% 计算实验数据的去趋势切除局部平均十则不与实验相差，不影响整体形状
% 第一点偏移量（拦截至 t=0）保存不变，只去掉随时间的线性漂移
exp_fam_d = exp_fam - polyval(p_drift_fam, exp_time) + p_drift_fam(2);
exp_tye_d = exp_tye - polyval(p_drift_tye, exp_time) + p_drift_tye(2);
exp_cy5_d = exp_cy5 - polyval(p_drift_cy5, exp_time) + p_drift_cy5(2);

% 将去趋实验数据和各通道多项式系数一起传给目标函数
objective = @(x) compute_sim_mse(x, exp_time, exp_fam_d, exp_tye_d, exp_cy5_d, ...
                                  p_drift_fam, p_drift_tye, p_drift_cy5, lb, ub);

% 运行优化器
options = optimset( ...
    'MaxIter',   2000,  ...   % 最大迭代次数
    'MaxFunEvals', 20000, ... % 最大函数评估次数
    'TolX',     1e-6,   ...  % 参数收敛阈值
    'TolFun',   1e-8,   ...  % 函数值收敛阈值
    'Display',  'iter'  ...  % 显示迭代过程
);

[x_opt, mse_opt, exitflag, output] = fminsearch(objective, x0, options);

% 将优化结果裁剪到物理边界内（fminsearch不做边界约束，依靠惩罚函数）
x_opt = max(lb, min(ub, x_opt));

fprintf('\n--- 优化完成! ---\n');
fprintf('退出状态: %d  (1=收敛, 0=未收敛)\n', exitflag);
fprintf('迭代次数: %d  函数评估: %d\n', output.iterations, output.funcCount);
fprintf('最终 MSE: %.6e\n\n', mse_opt);

%% ============================================================
%  5. 打印参数对比（优化前 vs 优化后）
%% ============================================================

param_names = {'e_b', 'e_b_azo_trans', 'e_b_azo_cis', 'k0', 'k_mig', 'drt_z', 'drt_s'};

fprintf('=== 参数对比 ===\n');
fprintf('%-18s  %12s  %12s  %10s\n', '参数', 'ML初始值', '优化后值', '变化');
fprintf('%s\n', repmat('-', 1, 58));
for i = 1:7
    delta = (x_opt(i) - x0(i)) / abs(x0(i)) * 100;
    fprintf('%-18s  %12.4e  %12.4e  %+8.2f%%\n', param_names{i}, x0(i), x_opt(i), delta);
end
fprintf('%s\n\n', repmat('=', 1, 58));

%% ============================================================
%  6. 将优化后的参数写入 optimized_params.txt
%% ============================================================

output_file = 'optimized_params.txt';
fid = fopen(output_file, 'w');
fprintf(fid, 'e_b=%.16f\n',             x_opt(1));
fprintf(fid, 'e_b_azo_trans=%.16f\n',   x_opt(2));
fprintf(fid, 'e_b_azo_cis=%.16f\n',     x_opt(3));
fprintf(fid, 'k0=%.16e\n',              x_opt(4));
fprintf(fid, 'k_mig=%.16f\n',           x_opt(5));
fprintf(fid, 'drt_z=%.16f\n',           x_opt(6));
fprintf(fid, 'drt_s=%.16f\n',           x_opt(7));
fprintf(fid, 'END_OF_PARAMS=1\n');
fclose(fid);

fprintf('优化后的参数已保存至: %s\n\n', output_file);

%% ============================================================
%  7. 用优化后参数再跑一次完整仿真并绘图
%% ============================================================

fprintf('--- 4. 用优化参数绘制最终对比图 ---\n');
[result_signal] = run_simulation(x_opt);

sim_time = result_signal(:, 1);
sim_fam  = result_signal(:, 2);
sim_tye  = result_signal(:, 3);
sim_cy5  = result_signal(:, 4);

% 插值计算 RMSE
exp_fam_interp = interp1(exp_time, exp_fam, sim_time, 'linear', 'extrap');
exp_tye_interp = interp1(exp_time, exp_tye, sim_time, 'linear', 'extrap');
exp_cy5_interp = interp1(exp_time, exp_cy5, sim_time, 'linear', 'extrap');

rmse_fam  = sqrt(mean((sim_fam - exp_fam_interp).^2, 'omitnan'));
rmse_tye  = sqrt(mean((sim_tye - exp_tye_interp).^2, 'omitnan'));
rmse_cy5  = sqrt(mean((sim_cy5 - exp_cy5_interp).^2, 'omitnan'));
rmse_init = 0;  % 已用 mse_opt 替代，不再重新跑初始仿真（节省时间）

fprintf('\n=== 拟合误差 RMSE ===\n');
fprintf('FAM RMSE: %.6f\n', rmse_fam);
fprintf('TYE RMSE: %.6f\n', rmse_tye);
fprintf('CY5 RMSE: %.6f\n', rmse_cy5);
fprintf('平均 RMSE: %.6f\n', (rmse_fam+rmse_tye+rmse_cy5)/3);

figure('Position', [100, 100, 1200, 800]);
subplot(3, 1, 1);
plot(sim_time, sim_fam, 'r-', 'LineWidth', 2, 'DisplayName', 'Optimized Sim'); hold on;
scatter(exp_time, exp_fam, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Experimental');
xlabel('Time (min)'); ylabel('FAM Signal');
title(sprintf('FAM: Optimized Simulation vs Experimental  (RMSE=%.4f)', rmse_fam), 'FontWeight', 'bold');
legend('Location', 'best'); grid on; hold off;

subplot(3, 1, 2);
plot(sim_time, sim_tye, 'r-', 'LineWidth', 2, 'DisplayName', 'Optimized Sim'); hold on;
scatter(exp_time, exp_tye, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Experimental');
xlabel('Time (min)'); ylabel('TYE Signal');
title(sprintf('TYE: Optimized Simulation vs Experimental  (RMSE=%.4f)', rmse_tye), 'FontWeight', 'bold');
legend('Location', 'best'); grid on; hold off;

subplot(3, 1, 3);
plot(sim_time, sim_cy5, 'r-', 'LineWidth', 2, 'DisplayName', 'Optimized Sim'); hold on;
scatter(exp_time, exp_cy5, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Experimental');
xlabel('Time (min)'); ylabel('CY5 Signal');
title(sprintf('CY5: Optimized Simulation vs Experimental  (RMSE=%.4f)', rmse_cy5), 'FontWeight', 'bold');
legend('Location', 'best'); grid on; hold off;

saveas(gcf, 'optimized_vs_experimental.png');
saveas(gcf, 'optimized_vs_experimental.fig');
fprintf('\n对比图已保存为 optimized_vs_experimental.png/.fig\n');
fprintf('程序运行完成!\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ============================================================
%  辅助函数 1: 计算物理仿真 MSE（供 fminsearch 调用）
%% ============================================================

function mse = compute_sim_mse(x, exp_time, exp_fam_d, exp_tye_d, exp_cy5_d, ...
                               p_drift_fam, p_drift_tye, p_drift_cy5, lb, ub)
    PENALTY = 1e6;
    violation = sum(max(0, lb - x).^2) + sum(max(0, x - ub).^2);
    if violation > 0
        mse = PENALTY * (1 + violation);
        return;
    end
    if x(4) <= 0 || x(5) <= 0 || x(6) <= 0 || x(7) <= 0
        mse = PENALTY;
        return;
    end
    try
        result_signal = run_simulation(x);
    catch
        mse = PENALTY;
        return;
    end
    sim_time = result_signal(:, 1);
    sim_fam  = result_signal(:, 2);
    sim_tye  = result_signal(:, 3);
    sim_cy5  = result_signal(:, 4);

    % 插值到实验时间点
    sim_fam_interp = interp1(sim_time, sim_fam, exp_time, 'linear', 'extrap');
    sim_tye_interp = interp1(sim_time, sim_tye, exp_time, 'linear', 'extrap');
    sim_cy5_interp = interp1(sim_time, sim_cy5, exp_time, 'linear', 'extrap');

    % 对仿真信号去除同样的线性漂移（t=0 截距保持不变）
    % 这样 diff = (sim_detrended) - (exp_detrended)，漂移正好相消
    sim_fam_d = sim_fam_interp - polyval(p_drift_fam, exp_time) + p_drift_fam(2);
    sim_tye_d = sim_tye_interp - polyval(p_drift_tye, exp_time) + p_drift_tye(2);
    sim_cy5_d = sim_cy5_interp - polyval(p_drift_cy5, exp_time) + p_drift_cy5(2);

    % Huber Loss（对去趋势后的信号计算）
    delta = 0.015;
    diff_fam = sim_fam_d - exp_fam_d;
    diff_tye = sim_tye_d - exp_tye_d;
    diff_cy5 = sim_cy5_d - exp_cy5_d;

    huber = @(d) mean( ...
        (abs(d) <= delta) .* (0.5 .* d.^2) + ...
        (abs(d)  > delta) .* (delta .* (abs(d) - 0.5*delta)), ...
        'omitnan');

    hl_fam = huber(diff_fam);
    hl_tye = huber(diff_tye);
    hl_cy5 = huber(diff_cy5);

    % 加权 Huber Loss（CY5 ×2）
    mse = (hl_fam + hl_tye + 2.0 * hl_cy5) / 4.0;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ============================================================
%  辅助函数 2: 运行完整物理仿真（从 verify.m 提取的核心代码）
%% ============================================================

function result_signal = run_simulation(x)
    % 参数解包
    % x = [E_b, E_b_azo_trans, E_b_azo_cis, k0, k_mig, drt_z, drt_s]
    E_b           = x(1);
    E_b_azo_trans = x(2);
    E_b_azo_cis   = x(3);
    k0            = x(4);
    k_mig         = x(5);
    drt_z         = x(6);
    drt_s         = x(7);

    % 力学常数
    kBT   = 4.14;
    lp_s  = 0.75;
    lc_s  = 0.7;
    lc_d  = 0.34;
    dE_TYE = -1.55;

    % 结构参数
    n_D1 = 10; n_D2 = 10; n_S1 = 4; n_gray = 10;
    n_hairpin_1 = 8; n_hairpin_2 = 8;
    n_azo_1 = 3; n_azo_2 = 3;
    n_T_hairpin_1 = 3; n_T_hairpin_2 = 2;
    n_track_1 = 15; n_track_2 = 55;

    % -----------------------------------------------------------------
    % 自由能计算（与 verify.m 完全相同）
    % -----------------------------------------------------------------
    E_shear_foot = 100;
    for i = 0:n_D2
        n_D2_detach = i;
        E_b_shear = E_b * (n_D1 + n_D2 - n_D2_detach);
        x_val = (n_track_1 * lc_d) / (lc_s * (2*n_D2_detach + n_D1));
        if x_val < 1
            E_shear = E_b_shear + (lc_s*(2*n_D2_detach+n_D1)) * x_val^2*(3-2*x_val)/4/(1-x_val);
        else
            E_shear = 1000;
        end
        if E_shear_foot > E_shear
            E_shear_foot = E_shear;
            n_shear_foot = n_D1 + n_D2 - n_D2_detach;
        end
    end
    E_zipper_foot = E_b * (n_D1 + n_D2);

    E_config_t(1) = E_zipper_foot; E_config_t(2) = E_shear_foot;
    E_config_c(1) = E_zipper_foot; E_config_c(2) = E_shear_foot;

    % States 3-6 (double foot binding)
    configs = {
        [n_track_1+n_track_2-2*n_gray], ...  % state 3/4: full track
        [n_track_1+n_track_2-2*n_gray], ...  % state 3/4
        [n_track_2-2*n_gray], ...             % state 5: one-track
        [2*n_track_1+n_track_2-2*n_gray]      % state 6: extended
    };
    foot_types = {
        [E_zipper_foot, E_zipper_foot], ...
        [E_shear_foot,  E_shear_foot ], ...
        [E_zipper_foot, E_shear_foot ], ...
        [E_zipper_foot, E_shear_foot ]
    };
    for st = 1:4
        E_state_min_t = 1000; f_state_min_t = 0.0;
        E_state_min_c = 1000; f_state_min_c = 0.0;
        track_len = configs{st};
        feet = foot_types{st};
        for i = 1:(n_hairpin_1 + n_hairpin_2)
            n_hairpin_open = i;
            if n_hairpin_open < n_hairpin_1
                x_val = (track_len * lc_d) / (n_hairpin_open * 2 * lc_s);
                n_chain = n_hairpin_open;
            elseif n_hairpin_open < n_hairpin_1 + n_hairpin_2
                x_val = (track_len * lc_d) / ((n_hairpin_open + n_T_hairpin_1) * 2 * lc_s);
                n_chain = n_hairpin_open + n_T_hairpin_1;
            else
                x_val = (track_len * lc_d) / ((n_hairpin_open + n_T_hairpin_1 + n_T_hairpin_2) * 2 * lc_s);
                n_chain = n_hairpin_open + n_T_hairpin_1 + n_T_hairpin_2;
            end
            if x_val < 1
                E_neck  = 2 * (n_chain*2*lc_s/lp_s) * x_val^2*(3-2*x_val)/4/(1-x_val);
                f_state = 2*kBT/lp_s * (x_val - 0.25 + (1-x_val)^-2/4);
            else
                E_neck = 1000; f_state = 1000;
            end
            E_state_t = E_neck + feet(1) + feet(2) - 2*n_hairpin_open*E_b_azo_trans;
            E_state_c = E_neck + feet(1) + feet(2) - 2*n_hairpin_open*E_b_azo_cis;
            if E_state_min_t > E_state_t
                E_state_min_t = E_state_t; f_state_min_t = f_state; n_state_open_t = i;
            end
            if E_state_min_c > E_state_c
                E_state_min_c = E_state_c; f_state_min_c = f_state; n_state_open_c = i;
            end
        end
        E_config_t(st+2) = E_state_min_t; f_config_t(st+2) = f_state_min_t;
        E_config_c(st+2) = E_state_min_c; f_config_c(st+2) = f_state_min_c;
    end

    % 扩展 6 states → 14 states（与 verify.m 完全相同）
    E_config_t_copy = E_config_t; E_config_c_copy = E_config_c;
    f_config_t_copy = f_config_t; f_config_c_copy = f_config_c;

    E_config_t(1:3)   = E_config_t_copy(1); E_config_t(4:6)   = E_config_t_copy(2);
    E_config_t(7:8)   = E_config_t_copy(3); E_config_t(9:10)  = E_config_t_copy(4);
    E_config_t(11:12) = E_config_t_copy(5); E_config_t(13:14) = E_config_t_copy(6);

    E_config_c(1:3)   = E_config_c_copy(1); E_config_c(4:6)   = E_config_c_copy(2);
    E_config_c(7:8)   = E_config_c_copy(3); E_config_c(9:10)  = E_config_c_copy(4);
    E_config_c(11:12) = E_config_c_copy(5); E_config_c(13:14) = E_config_c_copy(6);

    f_config_t(1:3)   = f_config_t_copy(1); f_config_t(4:6)   = f_config_t_copy(2);
    f_config_t(7:8)   = f_config_t_copy(3); f_config_t(9:10)  = f_config_t_copy(4);
    f_config_t(11:12) = f_config_t_copy(5); f_config_t(13:14) = f_config_t_copy(6);

    f_config_c(1:3)   = f_config_c_copy(1); f_config_c(4:6)   = f_config_c_copy(2);
    f_config_c(7:8)   = f_config_c_copy(3); f_config_c(9:10)  = f_config_c_copy(4);
    f_config_c(11:12) = f_config_c_copy(5); f_config_c(13:14) = f_config_c_copy(6);

    % TYE 能量修正（与 verify.m 相同）
    for idx = [1,4,7,9,11,13]
        E_config_t(idx) = E_config_t(idx) + dE_TYE;
        E_config_c(idx) = E_config_c(idx) + dE_TYE;
    end

    % -----------------------------------------------------------------
    % 速率矩阵（与 verify.m 完全相同）
    % -----------------------------------------------------------------
    k_trans = zeros(14,14);
    k_cis   = zeros(14,14);

    % trans: single-single
    k_trans(4,1)=k_mig; k_trans(5,2)=k_mig; k_trans(6,3)=k_mig;
    k_trans(1,4)=k_trans(4,1)*exp(E_config_t(1)-E_config_t(4));
    k_trans(2,5)=k_trans(5,2)*exp(E_config_t(2)-E_config_t(5));
    k_trans(3,6)=k_trans(6,3)*exp(E_config_t(3)-E_config_t(6));
    % trans: single-double
    k_trans(7,1) =k0*exp(f_config_t(7) *drt_z/kBT); k_trans(11,1)=k0*exp(f_config_t(11)*drt_s/kBT);
    k_trans(1,7) =k_trans(7,1) *exp(E_config_t(1)-E_config_t(7));
    k_trans(1,11)=k_trans(11,1)*exp(E_config_t(1)-E_config_t(11));
    k_trans(7,2) =k0*exp(f_config_t(7) *drt_z/kBT); k_trans(8,2) =k0*exp(f_config_t(8) *drt_z/kBT);
    k_trans(12,2)=k0*exp(f_config_t(12)*drt_s/kBT); k_trans(13,2)=k0*exp(f_config_t(13)*drt_s/kBT);
    k_trans(2,7) =k_trans(7,2) *exp(E_config_t(2)-E_config_t(7));
    k_trans(2,8) =k_trans(8,2) *exp(E_config_t(2)-E_config_t(8));
    k_trans(2,12)=k_trans(12,2)*exp(E_config_t(2)-E_config_t(12));
    k_trans(2,13)=k_trans(13,2)*exp(E_config_t(2)-E_config_t(13));
    k_trans(8,3) =k0*exp(f_config_t(8) *drt_z/kBT); k_trans(14,3)=k0*exp(f_config_t(14)*drt_s/kBT);
    k_trans(3,8) =k_trans(8,3) *exp(E_config_t(3)-E_config_t(8));
    k_trans(3,14)=k_trans(14,3)*exp(E_config_t(3)-E_config_t(14));
    k_trans(9,4) =k0*exp(f_config_t(9) *drt_s/kBT); k_trans(13,4)=k0*exp(f_config_t(13)*drt_z/kBT);
    k_trans(4,9) =k_trans(9,4) *exp(E_config_t(4)-E_config_t(9));
    k_trans(4,13)=k_trans(13,4)*exp(E_config_t(4)-E_config_t(13));
    k_trans(9,5) =k0*exp(f_config_t(9) *drt_s/kBT); k_trans(10,5)=k0*exp(f_config_t(10)*drt_s/kBT);
    k_trans(11,5)=k0*exp(f_config_t(11)*drt_z/kBT); k_trans(14,5)=k0*exp(f_config_t(14)*drt_z/kBT);
    k_trans(5,9) =k_trans(9,5) *exp(E_config_t(5)-E_config_t(9));
    k_trans(5,10)=k_trans(10,5)*exp(E_config_t(5)-E_config_t(10));
    k_trans(5,11)=k_trans(11,5)*exp(E_config_t(5)-E_config_t(11));
    k_trans(5,14)=k_trans(14,5)*exp(E_config_t(5)-E_config_t(14));
    k_trans(10,6)=k0*exp(f_config_t(10)*drt_s/kBT); k_trans(12,6)=k0*exp(f_config_t(12)*drt_z/kBT);
    k_trans(6,10)=k_trans(10,6)*exp(E_config_t(6)-E_config_t(10));
    k_trans(6,12)=k_trans(12,6)*exp(E_config_t(6)-E_config_t(12));
    % trans: double-double
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

    % cis: single-single
    k_cis(4,1)=k_mig; k_cis(5,2)=k_mig; k_cis(6,3)=k_mig;
    k_cis(1,4)=k_cis(4,1)*exp(E_config_c(1)-E_config_c(4));
    k_cis(2,5)=k_cis(5,2)*exp(E_config_c(2)-E_config_c(5));
    k_cis(3,6)=k_cis(6,3)*exp(E_config_c(3)-E_config_c(6));
    % cis: single-double
    k_cis(7,1) =k0*exp(f_config_c(7) *drt_z/kBT); k_cis(11,1)=k0*exp(f_config_c(11)*drt_s/kBT);
    k_cis(1,7) =k_cis(7,1) *exp(E_config_c(1)-E_config_c(7));
    k_cis(1,11)=k_cis(11,1)*exp(E_config_c(1)-E_config_c(11));
    k_cis(7,2) =k0*exp(f_config_c(7) *drt_z/kBT); k_cis(8,2) =k0*exp(f_config_c(8) *drt_z/kBT);
    k_cis(12,2)=k0*exp(f_config_c(12)*drt_s/kBT); k_cis(13,2)=k0*exp(f_config_c(13)*drt_s/kBT);
    k_cis(2,7) =k_cis(7,2) *exp(E_config_c(2)-E_config_c(7));
    k_cis(2,8) =k_cis(8,2) *exp(E_config_c(2)-E_config_c(8));
    k_cis(2,12)=k_cis(12,2)*exp(E_config_c(2)-E_config_c(12));
    k_cis(2,13)=k_cis(13,2)*exp(E_config_c(2)-E_config_c(13));
    k_cis(8,3) =k0*exp(f_config_c(8) *drt_z/kBT); k_cis(14,3)=k0*exp(f_config_c(14)*drt_s/kBT);
    k_cis(3,8) =k_cis(8,3) *exp(E_config_c(3)-E_config_c(8));
    k_cis(3,14)=k_cis(14,3)*exp(E_config_c(3)-E_config_c(14));
    k_cis(9,4) =k0*exp(f_config_c(9) *drt_s/kBT); k_cis(13,4)=k0*exp(f_config_c(13)*drt_z/kBT);
    k_cis(4,9) =k_cis(9,4) *exp(E_config_c(4)-E_config_c(9));
    k_cis(4,13)=k_cis(13,4)*exp(E_config_c(4)-E_config_c(13));
    k_cis(9,5) =k0*exp(f_config_c(9) *drt_s/kBT); k_cis(10,5)=k0*exp(f_config_c(10)*drt_s/kBT);
    k_cis(11,5)=k0*exp(f_config_c(11)*drt_z/kBT); k_cis(14,5)=k0*exp(f_config_c(14)*drt_z/kBT);
    k_cis(5,9) =k_cis(9,5) *exp(E_config_c(5)-E_config_c(9));
    k_cis(5,10)=k_cis(10,5)*exp(E_config_c(5)-E_config_c(10));
    k_cis(5,11)=k_cis(11,5)*exp(E_config_c(5)-E_config_c(11));
    k_cis(5,14)=k_cis(14,5)*exp(E_config_c(5)-E_config_c(14));
    k_cis(10,6)=k0*exp(f_config_c(10)*drt_s/kBT); k_cis(12,6)=k0*exp(f_config_c(12)*drt_z/kBT);
    k_cis(6,10)=k_cis(10,6)*exp(E_config_c(6)-E_config_c(10));
    k_cis(6,12)=k_cis(12,6)*exp(E_config_c(6)-E_config_c(12));
    % cis: double-double
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

    % -----------------------------------------------------------------
    % 时间步长（与 verify.m 相同，带安全限制）
    % -----------------------------------------------------------------
    mag_t = floor(log10(max(max(max(k_cis)), max(max(k_trans)))));
    dt = 1/10^mag_t/10;
    dt = max(1e-6, min(0.1, dt));  % 限制在 [1e-6, 0.1] 秒

    % -----------------------------------------------------------------
    % 构建传播矩阵
    % -----------------------------------------------------------------
    R_vis = zeros(14,14);
    R_UV  = zeros(14,14);
    for i = 1:14
        for j = 1:14
            if i == j
                R_vis(i,i) = 1; R_UV(i,i) = 1;
                for nn = 1:14
                    R_vis(i,i) = R_vis(i,i) - k_trans(i,nn)*dt;
                    R_UV(i,i)  = R_UV(i,i)  - k_cis(i,nn) *dt;
                end
            else
                R_vis(i,j) = k_trans(j,i)*dt;
                R_UV(i,j)  = k_cis(j,i) *dt;
            end
        end
    end

    % -----------------------------------------------------------------
    % 初始条件（与 verify.m 相同）
    % -----------------------------------------------------------------
    p_total = 0.945;
    p_config = zeros(14,1);
    p_config(11:12) = exp(-E_config_c(11:12) - 20);
    pp = sum(p_config);
    p_config(11:12) = p_config(11:12) / pp * p_total;

    % -----------------------------------------------------------------
    % 时间演化（130 分钟）
    % -----------------------------------------------------------------
    total_steps = floor(130*60/dt);
    result_signal = zeros(floor(total_steps/60)+2, 4);
    i_result = 1;

    for i = 0:total_steps
        if mod(floor(i*dt/60/10)+1, 2) == 1
            R_dyn = R_vis;
        else
            R_dyn = R_UV;
        end

        if i > 0
            p_config = R_dyn * p_config;
        end

        if mod(round(i*dt), 60) == 0
            signal_FAM = p_config(1)+p_config(2)+p_config(4)+p_config(5)+p_config(7)+p_config(11)+p_config(9)+p_config(13);
            signal_TYE = p_config(2)+p_config(3)+p_config(5)+p_config(6)+p_config(8)+p_config(12)+p_config(10)+p_config(14);
            signal_CY5 = p_config(1)+p_config(3)+p_config(4)+p_config(6)+0.09507;

            result_signal(i_result, 1) = i*dt/60;
            result_signal(i_result, 2) = signal_FAM;
            result_signal(i_result, 3) = signal_TYE;
            result_signal(i_result, 4) = signal_CY5;
            i_result = i_result + 1;
        end
    end

    result_signal = result_signal(1:i_result-1, :);
end
