function [data] = tree_bin_data(data, binsize, sum_length)
    trial_task_length = size(data, 2);
    rounded_length = floor(trial_task_length / binsize) * binsize;
    if binsize > 1
       binned_data = [];
       for bin = binsize:binsize:rounded_length
            bin_start = bin - binsize+ 1;
            binned_data = [binned_data sum(data(:, bin_start:bin), 2)];
       end
       data = binned_data;
    end
    if sum_length > 1
        data = movsum(data, [sum_length 0], 2);
    end
end