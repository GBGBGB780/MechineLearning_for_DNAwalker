clc
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%           Mechanics and kinetics of the winding DNA motor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%**************************************************************************
%                    READ PARAMETERS FROM INPUT FILE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 定义输入文件路径
input_file = 'matlab_input_params.txt';

% 检查文件是否存在
if ~exist(input_file, 'file')
    error('错误: 未找到输入文件 %s。请确保参数文件存在!', input_file);
end

fprintf('正在从 %s 读取参数...\n', input_file);

% 读取参数文件
fid = fopen(input_file, 'r');
params = struct();

while ~feof(fid)
    line = fgetl(fid);
    if contains(line, '=')
        parts = strsplit(line, '=');
        param_name = strtrim(parts{1});
        param_value = str2double(strtrim(parts{2}));
        
        % 跳过 END_OF_PARAMS
        if ~strcmp(param_name, 'END_OF_PARAMS')
            params.(param_name) = param_value;
            fprintf('  %s = %e\n', param_name, param_value);
        end
    end
end
fclose(fid);

fprintf('参数读取完成!\n\n');

%**************************************************************************
%                               mechanics parameters
kBT=4.14;            % nm*pN 

lp_s=0.75;           % nm persistence length of one nucleotide of single strand DNA
lc_s=0.7;            % nm contour length of one nucleotide of single strand DNA
lc_d=0.34;           % nm contour length of one base-pair of duplex DNA

% 必需参数列表
required_params = {'e_b', 'e_b_azo_trans', 'e_b_azo_cis', 'k0', 'k_mig', 'drt_z', 'drt_s'};

% 检查所有必需参数是否存在
fprintf('检查必需参数...\n');
for i = 1:length(required_params)
    param_name = required_params{i};
    if ~isfield(params, param_name)
        error('错误: 缺少必需参数 "%s"。请在输入文件中提供该参数!', param_name);
    end
end
fprintf('所有必需参数已找到!\n\n');

% 读取参数
E_b = params.e_b;
E_b_azo_trans = params.e_b_azo_trans;
E_b_azo_cis = params.e_b_azo_cis;

di_DNA=2;            % nm diameter of DNA helix

n_D1=10;             % number of nucleotide of D1
n_D2=10;             % number of nucleotide of D2
n_S1=4;              % number of nucleotide of S1
n_gray=10;           % number of base-pairs of the gray duplex
n_hairpin_1=8;       % number of base-pairs in the first segment of hairpin
n_hairpin_2=8;       % number of base-pairs in the second segment of hairpin
n_azo_1=3;           % half number of azo in the first segment of hairpin
n_azo_2=3;           % half number of azo in the second segment of hairpin
n_T_hairpin_1=3;     % half number of unpaired T between the first and second segments of the hairpin
n_T_hairpin_2=2;     % half number of unpaired T after the second segment of the hairpin

n_track_1=15;        % number of base-pairs of short segment of the track
n_track_2=55;        % number of base-pairs of long segment of the track

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
E_config_t(6)=E_state_min_t
f_config_t(6)=f_state_min_t
n_config_t(6)=n_state_open_t
E_config_c(6)=E_state_min_c
f_config_c(6)=f_state_min_c
n_config_c(6)=n_state_open_c



%**************************************************************************
%                              kinetics

%..........................................................................
%                         kinetics parameter

p_unbind_track=0.09507;

% 读取动力学参数（已在开头检查过必须存在）
k0 = params.k0;          % s-1 leg detachment rate at zero force
k_mig = params.k_mig;    % s-1 leg migration rate
drt_z = params.drt_z;    % nm  coupling distance for leg dissociation by unzipping
drt_s = params.drt_s;    % nm  coupling distance for leg dissociation by shearing

dE_TYE=-1.55;            % kBT

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

E_config_t
E_config_c
%..........................................................................
%                 transition rate (k) matrix

k_trans(14,14)=0;
k_cis(14,14)=0;

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

%k_trans(4,13)
%k_cis(4,13)

%max(max(k_cis))
k_cis(2,12);
k_cis(2,13);
k_cis(2,7);
k_cis(2,8);
k_cis(5,11);
%..........................................................................
%                     master equation R matrix

mag_t=floor(log10(max(max(max(k_cis)),max(max(k_trans)))));

dt=1/10^mag_t/10;

% 限制 dt 的最小值，防止循环次数过大
dt_min = 1e-6;  % 最小时间步长：1微秒
dt_max = 0.1;   % 最大时间步长：0.1秒

if dt < dt_min
    fprintf('警告: 计算得到的 dt (%.2e) 太小，已限制为最小值 %.2e\n', dt, dt_min);
    dt = dt_min;
elseif dt > dt_max
    fprintf('警告: 计算得到的 dt (%.2e) 太大，已限制为最大值 %.2e\n', dt, dt_max);
    dt = dt_max;
end

% 计算并显示循环次数
total_iterations = 130*60*1/dt;
fprintf('\n=== 模拟参数 ===\n');
fprintf('时间步长 dt: %.2e 秒\n', dt);
fprintf('模拟总时间: 130 分钟\n');
fprintf('总循环次数: %.2e\n', total_iterations);
fprintf('==================\n\n');

R_vis(14,14)=0;
R_UV(14,14)=0;

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
p_config(11:12)=p_config(11:12)/pp*p_total;

p_config=p_config';

%..........................................................................
%                               dynamics

% 采样间隔: 每秒记录一次 (与 gendata.m 一致)
save_interval_min = 1/60;  % 1秒 = 1/60 分钟
simu_time = 130;           % 总模拟时间 (分钟)
num_results = simu_time / save_interval_min + 1;  % 7801 个数据点
total_steps = round(simu_time * 60 / dt);

% 预分配数组
result_signal = zeros(num_results, 4);
result_p_s = zeros(num_results, 7);
result_p_d = zeros(num_results, 9);
result_b_fb = zeros(num_results, 3);
result_mig = zeros(num_results, 6);

i_result=1;
for i=0:1:total_steps
    current_time_min = i * dt / 60;
    
    if mod(floor(current_time_min/10)+1,2)==1
        R_dyn=R_vis;
        k_dyn=k_trans;
    else
        R_dyn=R_UV;
        k_dyn=k_cis;
    end
    
    if i==0
        dp14=0;
        dp25=0;
        dp36=0;
        dp511=0;
        dp212=0;
    else
        p_config_old=p_config;
        p_config=R_dyn*p_config;
        dp14=dp14+dt*(p_config_old(4)*k_dyn(4,1)-p_config_old(1)*k_dyn(1,4));
        dp25=dp25+dt*(p_config_old(5)*k_dyn(5,2)-p_config_old(2)*k_dyn(2,5));
        dp36=dp36+dt*(p_config_old(6)*k_dyn(6,3)-p_config_old(3)*k_dyn(3,6));
        dp511=dp511+dt*(p_config_old(11)*k_dyn(11,5)-p_config_old(5)*k_dyn(5,11));
        dp212=dp212+dt*(p_config_old(2)*k_dyn(2,12)-p_config_old(12)*k_dyn(12,2));
    end    
    
    if mod(floor(current_time_min/10)+1,2)==1
        b_forward=0;
        b_backward=0;
    else
        b_forward=b_forward+p_config(2)*k_cis(2,8)*dt-p_config(8)*k_cis(8,2)*dt+p_config(2)*k_cis(2,12)*dt-p_config(12)*k_cis(12,2)*dt+p_config(5)*k_cis(5,10)*dt-p_config(10)*k_cis(10,5)*dt+p_config(5)*k_cis(5,14)*dt-p_config(14)*k_cis(14,5)*dt;
        b_backward=b_backward+p_config(2)*k_cis(2,7)*dt-p_config(7)*k_cis(7,2)*dt+p_config(2)*k_cis(2,13)*dt-p_config(13)*k_cis(13,2)*dt+p_config(5)*k_cis(5,9)*dt-p_config(9)*k_cis(9,5)*dt+p_config(5)*k_cis(5,11)*dt-p_config(11)*k_cis(11,5)*dt;
    end
    
    % 每秒记录一次 (与 gendata.m 一致)
    if mod(current_time_min, save_interval_min) < (dt / 60)
        if i_result > num_results
            break;
        end
        signal_FAM=p_config(1)+p_config(2)+p_config(4)+p_config(5)+p_config(7)+p_config(11)+p_config(9)+p_config(13);
        signal_TYE=p_config(2)+p_config(3)+p_config(5)+p_config(6)+p_config(8)+p_config(12)+p_config(10)+p_config(14);
        signal_CY5=p_config(1)+p_config(3)+p_config(4)+p_config(6)+0.09507;
        
        result_signal(i_result,1)=current_time_min;
        result_signal(i_result,2)=signal_FAM;
        result_signal(i_result,3)=signal_TYE;
        result_signal(i_result,4)=signal_CY5;
        
        result_p_s(i_result,1)=current_time_min;
        result_p_s(i_result,2)=p_config(1);
        result_p_s(i_result,3)=p_config(2);
        result_p_s(i_result,4)=p_config(3);
        result_p_s(i_result,5)=p_config(4);
        result_p_s(i_result,6)=p_config(5);
        result_p_s(i_result,7)=p_config(6);
        
        result_p_d(i_result,1)=current_time_min;
        result_p_d(i_result,2)=p_config(7);
        result_p_d(i_result,3)=p_config(8);
        result_p_d(i_result,4)=p_config(9);
        result_p_d(i_result,5)=p_config(10);
        result_p_d(i_result,6)=p_config(11);
        result_p_d(i_result,7)=p_config(12);
        result_p_d(i_result,8)=p_config(13);
        result_p_d(i_result,9)=p_config(14);
        
        result_b_fb(i_result,1)=current_time_min;
        result_b_fb(i_result,2)=b_forward;
        result_b_fb(i_result,3)=b_backward;
        
        result_mig(i_result,1)=current_time_min;
        result_mig(i_result,2)=dp14;
        result_mig(i_result,3)=dp25;
        result_mig(i_result,4)=dp36;
        result_mig(i_result,5)=dp511;
        result_mig(i_result,6)=dp212;

        
        i_result=i_result+1;
        
    end
    
end
%p_config
result_signal_change(:,1)=result_signal(:,1);
result_signal_change(:,2)=-result_signal(:,2)+result_signal(1,2);
result_signal_change(:,3)=-result_signal(:,3)+result_signal(1,3);
result_signal_change(:,4)=-result_signal(:,4)+result_signal(1,4);
result_signal_change(:,5)=(result_signal_change(:,2)+result_signal_change(:,3)+result_signal_change(:,4))/3;
result_signal_change(:,6)=result_signal_change(:,2)-result_signal_change(:,5);
result_signal_change(:,7)=result_signal_change(:,3)-result_signal_change(:,5);
result_signal_change(:,8)=result_signal_change(:,4)-result_signal_change(:,5);

% 获取实际的数据点数量
num_data_points = size(result_signal, 1);

% 动态计算循环次数，避免数组越界
% 采样现在是每秒1个点，20分钟=1200个点，10分钟=600个点
% result_DD: 需要访问 i*1200+1，所以最大 i 满足 i*1200+1 <= num_data_points
max_i_DD = floor((num_data_points - 1) / 1200);
if max_i_DD > 0
    for i=1:1:max_i_DD
        result_DD(i,1)=i;
        result_DD(i,2)=(result_signal(i*1200+1,3)-result_signal(1,3))/result_signal(1,3)-(result_signal(i*1200+1,2)-result_signal(1,2))/result_signal(1,2);
        result_DD(i,3)=((result_signal(i*1200+1,4)-result_signal(1,4))/result_signal(1,4)+(result_signal(i*1200+1,3)-result_signal(1,3))/result_signal(1,3)+(result_signal(i*1200+1,2)-result_signal(1,2))/result_signal(1,2))/3;
    end
    fprintf('result_DD: 计算了 %d 个数据点\n', max_i_DD);
else
    fprintf('警告: result_signal 数据点不足，无法计算 result_DD\n');
    result_DD = [];
end

% result_bb: 需要访问 i*1200+1 和 i*1200-600+1，所以最大 i 满足 i*1200+1 <= num_data_points
max_i_bb = floor((num_data_points - 1) / 1200);
if max_i_bb > 0
    for i=1:1:max_i_bb
        result_bb(i,1)=i;
        result_bb(i,2)=(result_signal(i*1200+1,2)-result_signal(i*1200-600+1,2))/(result_signal(i*1200+1,3)-result_signal(i*1200-600+1,3));
    end
    fprintf('result_bb: 计算了 %d 个数据点\n', max_i_bb);
else
    fprintf('警告: result_signal 数据点不足，无法计算 result_bb\n');
    result_bb = [];
end

% result_bb_2: 需要访问 i*1200，所以最大 i 满足 i*1200 <= size(result_b_fb,1)
num_b_fb_points = size(result_b_fb, 1);
max_i_bb2 = floor(num_b_fb_points / 1200);
if max_i_bb2 > 0
    for i=1:1:max_i_bb2
        result_bb_2(i,1)=i;
        result_bb_2(i,2)=result_b_fb(i*1200,2)/result_b_fb(i*1200,3);
    end
    fprintf('result_bb_2: 计算了 %d 个数据点\n', max_i_bb2);
else
    fprintf('警告: result_b_fb 数据点不足，无法计算 result_bb_2\n');
    result_bb_2 = [];
end

save result_signal.txt result_signal -ASCII
save result_signal_change.txt result_signal_change -ASCII
save result_p_s.txt result_p_s -ASCII
save result_p_d.txt result_p_d -ASCII
save result_DD.txt result_DD -ASCII
save result_bb.txt result_bb -ASCII
save result_b_fb.txt result_b_fb -ASCII
save result_bb_2.txt result_bb_2 -ASCII
save result_mig.txt result_mig -ASCII

%**************************************************************************
%                  READ EXPERIMENTAL DATA AND PLOT COMPARISON
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n正在读取实验数据并生成对比图...\n');

% 定义实验数据文件路径
exp_data_file = 'Fig3a_fitting.xlsx';

% 检查文件是否存在
if exist(exp_data_file, 'file')
    % 读取实验数据
    exp_data = readtable(exp_data_file);
    
    % 显示实际的列名（用于调试）
    fprintf('Excel文件的列名:\n');
    disp(exp_data.Properties.VariableNames);
    
    % 提取列数据 - MATLAB会自动将特殊字符转换为下划线
    exp_time = exp_data.Time;  % 时间
    % 尝试不同的可能列名
    if ismember('FAM_FAMT__', exp_data.Properties.VariableNames)
        exp_fam = exp_data.FAM_FAMT__;
    elseif ismember('x_FAM_FAMT__', exp_data.Properties.VariableNames)
        exp_fam = exp_data.x_FAM_FAMT__;
    else
        exp_fam = exp_data{:, 2};  % 使用索引
    end
    
    if ismember('TYE_TYET__', exp_data.Properties.VariableNames)
        exp_tye = exp_data.TYE_TYET__;
    elseif ismember('x_TYE_TYET__', exp_data.Properties.VariableNames)
        exp_tye = exp_data.x_TYE_TYET__;
    else
        exp_tye = exp_data{:, 3};  % 使用索引
    end
    
    if ismember('CY5_CY5T_m_', exp_data.Properties.VariableNames)
        exp_cy5 = exp_data.CY5_CY5T_m_;
    elseif ismember('x_CY5_CY5T_m_', exp_data.Properties.VariableNames)
        exp_cy5 = exp_data.x_CY5_CY5T_m_;
    else
        exp_cy5 = exp_data{:, 4};  % 使用索引
    end
    
    % 提取模拟结果
    sim_time = result_signal(:, 1);  % 模拟时间
    sim_fam = result_signal(:, 2);   % FAM 模拟结果
    sim_tye = result_signal(:, 3);   % TYE 模拟结果
    sim_cy5 = result_signal(:, 4);   % CY5 模拟结果
    
    % 创建对比图
    figure('Position', [100, 100, 1200, 800]);
    
    % 绘制 FAM 对比
    subplot(3, 1, 1);
    plot(sim_time, sim_fam, 'r-', 'LineWidth', 2, 'DisplayName', 'Simulation');
    hold on;
    scatter(exp_time, exp_fam, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Experimental');
    xlabel('Time (min)', 'FontSize', 12);
    ylabel('FAM Signal', 'FontSize', 12);
    title('FAM: Simulation vs Experimental Data', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    hold off;
    
    % 绘制 TYE 对比
    subplot(3, 1, 2);
    plot(sim_time, sim_tye, 'r-', 'LineWidth', 2, 'DisplayName', 'Simulation');
    hold on;
    scatter(exp_time, exp_tye, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Experimental');
    xlabel('Time (min)', 'FontSize', 12);
    ylabel('TYE Signal', 'FontSize', 12);
    title('TYE: Simulation vs Experimental Data', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    hold off;
    
    % 绘制 CY5 对比
    subplot(3, 1, 3);
    plot(sim_time, sim_cy5, 'r-', 'LineWidth', 2, 'DisplayName', 'Simulation');
    hold on;
    scatter(exp_time, exp_cy5, 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Experimental');
    xlabel('Time (min)', 'FontSize', 12);
    ylabel('CY5 Signal', 'FontSize', 12);
    title('CY5: Simulation vs Experimental Data', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 10);
    grid on;
    hold off;
    
    % 保存图片
    saveas(gcf, 'simulation_vs_experimental.png');
    saveas(gcf, 'simulation_vs_experimental.fig');
    
    fprintf('对比图已保存为 simulation_vs_experimental.png 和 .fig\n');
    
    % 计算并显示拟合误差（均方根误差）
    % 需要对实验数据进行插值到模拟时间点
    exp_fam_interp = interp1(exp_time, exp_fam, sim_time, 'linear', 'extrap');
    exp_tye_interp = interp1(exp_time, exp_tye, sim_time, 'linear', 'extrap');
    exp_cy5_interp = interp1(exp_time, exp_cy5, sim_time, 'linear', 'extrap');
    
    rmse_fam = sqrt(mean((sim_fam - exp_fam_interp).^2));
    rmse_tye = sqrt(mean((sim_tye - exp_tye_interp).^2));
    rmse_cy5 = sqrt(mean((sim_cy5 - exp_cy5_interp).^2));
    rmse_total = (rmse_fam + rmse_tye + rmse_cy5) / 3;
    
    fprintf('\n=== 拟合误差分析 (RMSE) ===\n');
    fprintf('FAM RMSE: %.6f\n', rmse_fam);
    fprintf('TYE RMSE: %.6f\n', rmse_tye);
    fprintf('CY5 RMSE: %.6f\n', rmse_cy5);
    fprintf('Average RMSE: %.6f\n', rmse_total);
    fprintf('============================\n\n');
    
else
    fprintf('警告: 未找到实验数据文件 %s，无法生成对比图。\n', exp_data_file);
end

fprintf('\n程序运行完成!\n');

