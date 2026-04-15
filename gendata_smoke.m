clc;

if ~exist('results', 'dir')
    mkdir('results');
end

DNAWALKER_GENDATA_KEEP_WORKSPACE = true;
target_num_samples = 20;
initial_sample_ratio = 2.0;
simu_time = 130;
output_filename = fullfile('results', 'training_dataset_smoke.mat');
MAX_BATCH_SIZE = 20;
max_rounds = 10;
min_dt_threshold = 1.2e-5;
parpool_workers = 2;

run(fullfile(fileparts(mfilename('fullpath')), 'gendata.m'));
