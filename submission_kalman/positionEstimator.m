function [x, y, modelParameters] = positionEstimator(test_data, modelParameters, target_id)

%%% Uncomment if you want to use the angle decoder (right now it just assumes the correct position)
%     if size(test_data.spikes,2) == 320
%         modelParameters.estimated_angle  = 0;
%     end
%     
%     % Getting Firing rates estimate 
%     decoder = modelParameters.Pop_Vec;
%     times = sum(test_data.spikes,2); % number of spikes
%     
%      if size(test_data.spikes,2) <= 320
%         target_id = estimateReachingAngle_Classifier(decoder,times); % estimated reaching angle
%     else
%         target_id = modelParameters.estimated_angle;
%     end
%     
%     
%     T = size(test_data.spikes,2);
%     x = test_data.startHandPos(1);
%     y = test_data.startHandPos(2);
%     decoding_time = T - 300;
%     %x = modelParameters.Start_Pos(target_id).average(1);
%     %y = modelParameters.Start_Pos(target_id).average(2);
%     %x = x0;
%     %y = y0;
% 
%                
%     for t = 1:decoding_time
%         if t > size(modelParameters.Vel(target_id).average,2)
%             % do nothing if end of the signal was reached
%         else
%             x = x + modelParameters.Vel(target_id).average(1,t);
%             y = y + modelParameters.Vel(target_id).average(2,t);
%         end
%     end
%     
%     
%     modelParameters.estimated_angle = target_id;
    T = size(test_data.spikes,2);
    if T == 320
        modelParameters.distance = 500;
        modelParameters.curr_idx = 1;
    end
    
    [x, y] = kalman_prediction(test_data, modelParameters, target_id);
    distance = sqrt((x-modelParameters.avg_endPos(target_id).average(1,1))^2 + (y - modelParameters.avg_endPos(target_id).average(2,1))^2);
    modelParameters.distance = [modelParameters.distance, distance];
    
    % Modify the end x and y position if the decoder starts to move away
    % from the average end position
    % Check how much the current possition differs from the end position
    final_idx = size(modelParameters.distance,2);
    if final_idx > 1
        diff = modelParameters.distance(final_idx) - modelParameters.distance(final_idx-1);
    end
    
    % If the difference 
    if diff > 0
        
        x = test_data.decodedHandPos(1, modelParameters.curr_idx);
        y = test_data.decodedHandPos(2, modelParameters.curr_idx);
    else
        modelParameters.curr_idx = final_idx - 1;
    end
end

function [x_prediction, y_prediction] = kalman_prediction(td, modelParameters, target_id)
    filter = modelParameters.kalmanFit{target_id};
    experiment = prediction_data_formatting(td);
    
    bin_size = filter.bin_size;
    cell_list = filter.cell_list + 1;
    end_time_idx = size(experiment.spikes{1}, 2);
    start_time = 0;
    end_time = end_time_idx*bin_size;
    

    num_bins = floor( (end_time - start_time) / bin_size);
    num_cells = size(cell_list,1);
    
    % get the initial positions
    if isempty( td.decodedHandPos) 
        real_xpos = td.startHandPos(1,1);
        real_ypos = td.startHandPos(2,1);
        dx = 0;
        dy = 0;
    elseif size(td.decodedHandPos, 2) == 1
        % Check if this is true
        real_xpos = td.decodedHandPos(1,end);
        real_ypos = td.startHandPos(2,end);
        dx = td.decodedHandPos(1,end) - td.startHandPos(1,1);
        dy = td.decodedHandPos(2,end) - td.startHandPos(2,1);
    else
        % Check if this is true
        real_xpos = td.decodedHandPos(1,end);
        real_ypos = td.startHandPos(2,end);
        dx = td.decodedHandPos(1,end) - td.decodedHandPos(1,end-1);
        dy = td.decodedHandPos(2,end) - td.decodedHandPos(2,end-1);
    end
    
    % Get a nicely formatted matrix of (cells,bins) 
    %[unformatted_response,R] = make_response_matrix(num_cells,num_bins,experiment,cell_list,bin_size,start_time,0);
    unformatted_response = experiment.spikes{1}(cell_list(:,1),:);
    % Now I need the starting x,y,dx,dy

    cur_x_index = 1;
    cur_y_index = 1;
    
    % find the indices in the 'real' x and y arrays corresponding to the first
    % time point
    while(cur_x_index < size(real_xpos,1) & real_xpos(cur_x_index,1) < start_time)
        cur_x_index = cur_x_index + 1;
    end

    while(cur_y_index < size(real_ypos,1) & real_ypos(cur_y_index,1) < start_time)
        cur_y_index = cur_y_index + 1;
    end

    x = real_xpos;
    y = real_ypos;
    
    state = [(x - filter.center(1)) (y - filter.center(2)) dx dy]';
    state_m = state;
    
    x_predictions = zeros(num_bins,1);
    y_predictions = zeros(num_bins,1);
    window_times = zeros(num_bins,1);
    
    x_predictions(1) = x;
    y_predictions(1) = y;
    
    % Initial values for state variables
    P_m = zeros(4,4,num_bins);
    P = zeros(4,4,num_bins);
    K = zeros(4,num_cells,num_bins);

    A = filter.A;
    W = filter.W;
    H = filter.H;
    Q = filter.Q;

    step = 2;
    
  
    cur_time = start_time;
    window_times(1) = start_time + filter.lag*bin_size;

    while(step <= num_bins)

        % prior estimation 
        P_m(:,:,step) = A*P(:,:,step-1)*A'+W; 
        state_m(:,step) = A*state(:,step-1); 

        z = unformatted_response(:,step);

        % posterior estimation
        K(:,:,step) = P_m(:,:,step)*H'*inv(H*P_m(:,:,step)*H'+Q);       
        P(:,:,step) = (eye(4)-K(:,:,step)*H)*P_m(:,:,step);   
        state(:,step) = state_m(:,step)+K(:,:,step)*(z-H*state_m(:,step));  

        cur_time = cur_time + bin_size;
        window_times(step) = cur_time;

        % Here we account for the fact that the filter was built to imply some lag between kinematics
        % and spikes...
        window_times(step) = window_times(step) + filter.lag*bin_size;

        if (mod(step,100) == 0)
            fprintf(1,'.');
        end
        if (mod(step,60*100) == 0)
            fprintf(1,'\n');
        end

        x = state(1,step)+filter.center(1); % make X prediction
        y = state(2,step)+filter.center(2); % make Y prediction

        x_predictions(step) = x;
        y_predictions(step) = y;
        step = step + 1;
    end
    x_prediction = x_predictions(end);
    y_prediction = y_predictions(end);
end

function [formatted] = prediction_data_formatting(td)
    [N M] = size(td);
    for m = 1:M
        samp_count = 0;
        for n = 1:N
            for t = 20:20:length(td(n,m).spikes)
                samp_count = samp_count + 1;
                spikes{m}(:,samp_count) = sum(td(n,m).spikes(:,t-19:t),2);
            end
        end
        spike_count{m}=sum(spikes{m},2);
       
    end
    formatted.spikes = spikes;
    formatted.spike_count = spike_count;
end
