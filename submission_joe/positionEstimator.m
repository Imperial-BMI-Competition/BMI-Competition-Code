function [x, y, modelParameters] = positionEstimator(test_data, modelParameters) 
    
    if size(test_data.spikes,2) == 320
        modelParameters.estimated_angle = 0;
    end
    
    % Getting Firing rates estimate 
    decoder = modelParameters.Pop_Vec;
    times = sum(test_data.spikes,2); % number of spikes
    
     if size(test_data.spikes,2) <= 320
        target_id = estimateReachingAngle_Classifier(decoder,times); % estimated reaching angle
    else
        target_id = modelParameters.estimated_angle;
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
    traj_done = traj_done^0.8;
    
    x = (x_relative_avg * traj_done) + (x_true_avg * (1  - traj_done));
    y = (y_relative_avg * traj_done) + (y_true_avg * (1 - traj_done));
    
    modelParameters.estimated_angle = target_id;
end
