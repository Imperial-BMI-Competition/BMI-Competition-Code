function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
    
    data_length = size(test_data.spikes, 2);
    if data_length <= 320
        modelParameters.tree_estimated_angles = [];
        modelParameters.init_estimated_angles = [];
        modelParameters.init_estimated_angle = 0;
        modelParameters.estimated_angle = 0;
        modelParameters.mid_estimated_angles = [];
        modelParameters.mid_estimated_angle = 0;
    end
    
    if modelParameters.tree_freq ~= 0
        if mod(data_length - 320, modelParameters.tree_freq) == 0 
            tree_data = sum(test_data.spikes(:, data_length-modelParameters.tree_binsize:data_length),2);
            pred = predict(modelParameters.tree, tree_data');
            modelParameters.tree_estimated_angles = [modelParameters.tree_estimated_angles, pred];
        end
    end

    if data_length <= 320
        decoder = modelParameters.Pop_Vec;
        
        firing_rates = sum(test_data.spikes(:,1:modelParameters.init_idx),2); % number of spikes
        total_n_spikes = sum(firing_rates);
        
        F_test = firing_rates;
        if modelParameters.time_norm
            F_test = F_test./modelParameters.init_idx;
        end
        if modelParameters.max_norm
            F_test = F_test./total_n_spikes;
        end
        if modelParameters.mean_adjust
            F_test = F_test - modelParameters.avg_rate';
        end

        angles = bagged_estimateReachingAngle_Classifier(decoder, F_test'); % estimated reaching angle
        modelParameters.init_estimated_angles = angles;
        target_id = circular_mean(angles);
        modelParameters.init_estimated_angle = target_id;
        modelParameters.estimated_angle = target_id;
    else
        if data_length >= modelParameters.mid_idx
            if data_length == modelParameters.mid_idx
                decoder = modelParameters.Pop_Vec_Mid;
                firing_rates = sum(test_data.spikes(:,1:data_length),2); % number of spikes
                total_n_spikes = sum(firing_rates);                
                F_test = firing_rates;
                if modelParameters.time_norm
                    F_test = F_test./modelParameters.init_idx;
                end
                if modelParameters.max_norm
                    F_test = F_test./total_n_spikes;
                end
                if modelParameters.mean_adjust
                    F_test = F_test - modelParameters.avg_rate';
                end
                
                angles = bagged_estimateReachingAngle_Classifier(decoder, F_test'); % estimated reaching angle
                target_id = circular_mean(angles);
                modelParameters.mid_estimated_angles = angles;
                modelParameters.mid_estimated_angle = target_id;
                est_angles = [modelParameters.init_estimated_angles, modelParameters.mid_estimated_angles, modelParameters.tree_estimated_angles];
                modelParameters.estimated_angle = circular_mean(est_angles);
            end
        end
        target_id = modelParameters.estimated_angle;
    end
    
    decoding_time = data_length - 300;

    s = size(modelParameters.Vel(target_id).average, 2);
    if decoding_time > s
        decoding_time = s;
    end
    x_relative_avg = test_data.startHandPos(1) + modelParameters.Vel(target_id).average_cumsum(1, decoding_time);
    y_relative_avg = test_data.startHandPos(2) + modelParameters.Vel(target_id).average_cumsum(2, decoding_time);
    
    x_true_avg = modelParameters.Vel(target_id).avg_start_pos(1) + modelParameters.Vel(target_id).average_cumsum(1, decoding_time);
    y_true_avg = modelParameters.Vel(target_id).avg_start_pos(2) + modelParameters.Vel(target_id).average_cumsum(2, decoding_time);
    
    
    traj_done = (s - decoding_time) / (s-20);
    
    traj_done = traj_done^modelParameters.w;
    
    x = (x_relative_avg * traj_done) + (x_true_avg * (1  - traj_done));
    y = (y_relative_avg * traj_done) + (y_true_avg * (1 - traj_done));
end
