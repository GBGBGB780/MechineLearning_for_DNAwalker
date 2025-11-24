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
num_samples = 15000;      % 要生成的样本总数
simu_time = 130    % 模拟时间为130min
num_time_points = simu_time * 60 + 1;     % 标准化的时间点数量，总时间130min，因此7800点（秒），加上t=0，所以7801
output_filename = 'training_dataset.mat';

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

% X: (10000, 100, 3) 输出曲线矩阵
% (样本, 时间点, 曲线索引)
% 索引 1: FAM, 2: TYE, 3: CY5
X = zeros(num_samples, num_time_points, 3);

% 为了在 parfor 中高效传递，创建 fixed_params 的副本
fp = fixed_params;

X_local = cell(num_samples, 1);

num_r=0
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
    [time, fam, tye, cy5] = run_dna_motor_simulation(current_params, fp);
    % 将结果放入 cell 中（每个 cell 是一个 num_time_points x 3 矩阵）
    X_local{i} = [fam, tye, cy5]';

    if mod(i, 1) == 0
        num_r=num_r+1
        fprintf('已完成模拟 %d / %d\n', num_r, num_samples);
    end
end

% 4. 合并 cell 为三维矩阵
X = cat(3, X_local{:});
X = permute(X, [3,1,2]); % 调整维度，使其成为 (num_samples, num_time_points, 3)

disp('...所有并行模拟已完成。');

% 关闭并行池
% delete(gcp('nocreate'));
% disp('并行池已关闭。');

%% 6. 保存数据
disp(['正在将最终数据集保存到 ', output_filename, '...']);

% -v7.3 标志支持大于 2GB 的变量
save(output_filename, 'X', 'Y', 'param_names', '-v7.3');

toc; % 结束计时
disp('----------------------------------------------------');
disp('数据集生成完毕！');
disp(['文件已保存为: ', output_filename]);
disp('包含变量:');
disp('  X (输出): [10000, 100, 3] (样本, 时间点, 曲线)');
disp('  Y (输入): [10000, 7] (样本, 参数)');
disp('  param_names (标签): [1, 7] (参数名称)');
disp('----------------------------------------------------');


%% 7. 核心模拟函数 (本地函数)
% -------------------------------------------------------------------------
% ** 这是您提供的 .m 脚本的核心，已封装为一个函数 **
% -------------------------------------------------------------------------
function [time_out, fam_out, tye_out, cy5_out] = run_dna_motor_simulation(sim_params, fixed_params)
    % 该函数在 parfor 循环的每个 worker 上独立运行

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
    simu_time = 130
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
disp('模拟完成');
exit;