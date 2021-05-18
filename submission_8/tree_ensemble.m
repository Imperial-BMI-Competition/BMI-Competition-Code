function [model] = tree_ensemble(trainingData, filter_length, sum_length, binsize)
    
    x_filter = tree_get_filter(filter_length);
    
    for trial = 1:size(trainingData, 1)
       for task = 1:size(trainingData, 2)
           data = trainingData(trial, task).spikes;
           data = tree_bin_data(data, binsize, sum_length);
           data = tree_apply_filter(data, x_filter, filter_length);
           trainingData(trial, task).spikes_conv = data;
        end
    end
    
    x_data = [];
    y_ = [];
    
    start_floor_binsize = floor(300 / binsize);
    
    for trial = 1:size(trainingData, 1)
       for task = 1:size(trainingData, 2) 
           data = trainingData(trial, task).spikes_conv;
           l = size(data, 1);
           x_warmup = [];
           x_traj = [];
           for idx = 1:size(data, 2)             
%                [warmup_percent_done, traj_percent_done] = tree_calc_percent_done(idx, start_floor_binsize, l);
%                x_warmup = [x_warmup warmup_percent_done];
%                x_traj = [x_traj traj_percent_done];
               y_ = [y_ task];
           end
%            data = [data; x_warmup; x_traj];
           x_data = [x_data data];
       end
    end
    model = fitcensemble(x_data', y_');
end
