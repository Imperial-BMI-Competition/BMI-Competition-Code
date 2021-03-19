function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
    
    % Getting Firing rates estimate 
    decoder = modelParameters.Pop_Vec;
    times = sum(test_data.spikes,2); % number of spikes
    
    if size(test_data.spikes,2) <= 320
        target_id = bagged_estimateReachingAngle_Classifier(decoder,times); % estimated reaching angle
        modelParameters.init_estimated_angle = target_id;
        modelParameters.estimated_angles = [target_id target_id];
    else
        if modelParameters.retrain_sda
            spike_size = size(test_data.spikes, 2);
            lookback = 400;
            if mod(spike_size, 50) == 0
                lookback_start = spike_size - lookback;
                if lookback_start < 1
                    lookback_start = 1;
                end
                recent_spikes = test_data.spikes(:, lookback_start:end);
                recent_times = sum(recent_spikes, 2);
                times = recent_times;
                recent_target_id = bagged_estimateReachingAngle_Classifier(decoder, times);

                modelParameters.estimated_angles = [modelParameters.estimated_angles, recent_target_id];


    %             if modelParameters.init_estimated_angle ~= modelParameters.true_dir
    %                disp([num2str(size(test_data.spikes,2)), " ", num2str(recent_target_id)]); 
    %             end

            end
            target_id = circular_mean(modelParameters.estimated_angles);
        else
            target_id = modelParameters.estimated_angles(1);
        end
    end
  
%     target_id = modelParameters.true_dir;
    
    T = size(test_data.spikes,2);
    decoding_time = T - 300;
   
    s = size(modelParameters.Vel(target_id).average, 2);
    
    if decoding_time > s
        decoding_time = s;
    end
    x_relative_avg = test_data.startHandPos(1) + modelParameters.Vel(target_id).average_cumsum(1, decoding_time);
    y_relative_avg = test_data.startHandPos(2) + modelParameters.Vel(target_id).average_cumsum(2, decoding_time);

    x_true_avg = modelParameters.Vel(target_id).avg_start_pos(1) + modelParameters.Vel(target_id).average_cumsum(1, decoding_time);
    y_true_avg = modelParameters.Vel(target_id).avg_start_pos(2) + modelParameters.Vel(target_id).average_cumsum(2, decoding_time);
    
    traj_done = (s - decoding_time) / s;
    traj_done = traj_done^modelParameters.w;
    
    x = (x_relative_avg * traj_done) + (x_true_avg * (1  - traj_done));
    y = (y_relative_avg * traj_done) + (y_true_avg * (1 - traj_done));

end

