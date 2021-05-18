function [F_all, avg_n_spikes] = get_firing_ratio(data, test_start, test_end, hold_out, time_norm, mean_adjust, max_norm)
%GET_FIRING_RATES Summary of this function goes here
%   Detailed explanation goes here
    fsamp = 1000;
    angles = [30    70   110   150   190   230     310   350];

    %Get the dishcarge rates of all  neurons, angles and tasks
    for row = 1:size(data,1)
        for angle = 1:size(data,2)
            trial = data(row,angle).spikes;
            trial_length = size(trial, 2);
            n_inst = sum(trial(:,1:end),2);
            n_test_inst = sum(trial(:,test_start:test_end),2);
            if time_norm
                n_inst = n_inst / trial_length;
                n_test_inst = n_test_inst / (test_end - test_start);
            end
            n_spikes(:,angle,row) = n_inst;
            n_spikes_test(:,angle,row) = n_test_inst;
        end
    end
    
    % Check of neurons are directionally tuned 
    theta_radians = deg2rad(angles);
%     [x, y] = pol2cart(theta_radians, 1);  
%     unit_vectors = [x;y];
%     [r_max,s_a] = max(firing_rate,[],2);
 
     % Build Fetures Vector 
    F = []; F_ = [];
    y_true = []; y_true_ = [];

    F_test = [];
    y_true_test = [];
    
    avg_task_n_spikes = sum(n_spikes,3) / size(n_spikes,3);
    avg_n_spikes = sum(avg_task_n_spikes, 2) / 8;
    avg_n_spikes = avg_n_spikes';
    
    for i = 1: size(n_spikes,2)
        for j = 1:  size(n_spikes,3)
            f = n_spikes(:,i,j)';
            f_test = n_spikes_test(:,i,j)';
            
            % Training 
            if max_norm
                total_n_spikes = sum(n_spikes(:,i,j));
                f = f./total_n_spikes;
                avg_n_spikes = avg_n_spikes ./ total_n_spikes;
            end
            if mean_adjust
                f = f - avg_n_spikes;
            end
            F = [F;f]; 
            y_true = [y_true; i];

            % Testing - 320 msec only
            if max_norm
                total_n_spikes_test = sum(n_spikes_test(:,i,j));
                f_test = f_test./total_n_spikes_test;
                avg_n_spikes = avg_n_spikes ./ total_n_spikes;
            end
            if mean_adjust
                f_test = f_test - avg_n_spikes;
            end
            F_test = [F_test;f_test]; 
            y_true_test = [y_true_test; i];
            
        end
    end
    % Train classifier
    cv = cvpartition(size(F_test,1),'HoldOut', hold_out);
    idx = cv.test;
    F_all = [[F, y_true];[F_test(idx,:),y_true_test(idx,:)]];
end

