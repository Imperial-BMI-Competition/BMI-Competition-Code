

seeds = [20, 205, 1504, 2018];
for i = 1:10000
    batch_start = tic;
    rng(i);
    mp = getPara();
    [rmses, mrmse] = testManyFunc(seeds, mp);
    batch_end = toc(batch_start);
    fileID = fopen('results.txt','a');
    fprintf(fileID,'%d %d %d %d %f %f %f %s %d %d %d %d %d %d %d %d %f %f\n', mp.init_idx, mp.init_bag_size, mp.mid_idx, ...
            mp.mid_bag_size, mp.bag_data_split, mp.w, mp.hold_out, mp.discrimType, mp.pc_components, ...
            int8(mp.time_norm), int8(mp.mean_adjust), int8(mp.max_norm), ...
            mp.tree_filter_length, mp.tree_sum_length, mp.tree_binsize, mp.tree_freq, ...
            mrmse, batch_end);
    fclose(fileID);
end


function [modelParameters] = getPara()
    modelParameters.init_idx = 320;
    modelParameters.init_bag_size = randsample([1 2 4 8],1);
    modelParameters.mid_idx = randsample([340 360 380 400],1);
    modelParameters.mid_bag_size = randsample([1 2 4 8],1);
    modelParameters.bag_data_split = randsample([0.5 0.7 0.9 0.95 0.99],1);
    modelParameters.w = randsample([0. 0.1 0.5 0.9 1. 1.5 2.],1);
    modelParameters.hold_out = randsample([0.2 0.5 0.8],1);
    modelParameters.validateQD = false;
    if randsample(1:100, 1) == 1
        modelParameters.tree_filter_length = randsample([0 1 2],1); % 0, 1, 2
        modelParameters.tree_sum_length = randsample([0 1 2],1); % 0, 1, 2
        modelParameters.tree_binsize = randsample([240 360],1); % 40, 80, 120
        modelParameters.tree_freq = randsample([320 160],1);
    else
        modelParameters.tree_filter_length = 0;%randsample([0 1 2],1); % 0, 1, 2
        modelParameters.tree_sum_length = 0;%randsample([0 1 2],1); % 0, 1, 2
        modelParameters.tree_binsize = 0;%randsample([240 360],1); % 40, 80, 120
        modelParameters.tree_freq = 0; %randsample([0 160],1);
    end
    modelParameters.start_floor_binsize = floor(300 / modelParameters.tree_binsize);
    dis_idx = randsample([1 2 3],1);
    if dis_idx == 1
        modelParameters.discrimType = 'linear';
    elseif dis_idx == 2
        modelParameters.discrimType = 'quadratic';
    elseif dis_idx == 3
        modelParameters.discrimType = 'pseudoQuadratic';
    end
    modelParameters.pc_components = randsample([3 5 10 20 40],1);
    modelParameters.time_norm = randsample([true false],1);
    modelParameters.mean_adjust = randsample([true false],1);
    modelParameters.max_norm = randsample([true false],1);
end

