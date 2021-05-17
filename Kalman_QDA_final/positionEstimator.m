%%% Team Members: Joe Arrowsmith, Manfredi Castelli, Linnea Evanson, 
%%%               Joonsu Gha, Jon Skerlj
%%% BMI Spring 2021 (Update May 2021)

%%% Reference:
% Our implementation of the Kalman filter for neural decoding is based on
% that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
% The original implementation has previously been coded by Dan Morris 
%(http://dmorris.net/projects/neural_decoding.html#code)



function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)


    % Classification part of the decoder
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

            end
            target_id = circular_mean(modelParameters.estimated_angles);
        else
            target_id = modelParameters.estimated_angles(1);
        end
    end
    
    

    % Kalman filter part
    T = size(test_data.spikes,2);
    if T == 320
        modelParameters.distance = 500;
        modelParameters.curr_idx = 1;
    end
    
    % Predict x and y positions given classified target and trained Kalman
    % parameters
    [x, y] = kalman_prediction(test_data, modelParameters, target_id);
    
    % Find the distance between the estimation and average end position
    % for a given target of interest
    distance = sqrt((x-modelParameters.avg_endPos(target_id).average(1,1))^2 + (y - modelParameters.avg_endPos(target_id).average(2,1))^2);
    modelParameters.distance = [modelParameters.distance, distance];
    
    % Modify the end x and y position if the decoder starts to move away
    % from the average end position
    
    % Check how much the current possition differs from the end position
    final_idx = size(modelParameters.distance,2);
    % Find the difference between two time steps
    if final_idx > 1
        diff = modelParameters.distance(final_idx) - modelParameters.distance(final_idx-1);
    end
    
    
    if diff > 0
        % If the difference is bigger at later time step than before modify
        % x and y to be the same as before when the distance was smaller
        x = test_data.decodedHandPos(1, modelParameters.curr_idx);
        y = test_data.decodedHandPos(2, modelParameters.curr_idx);
    else
        modelParameters.curr_idx = final_idx - 1;
    end
end

% Prediction part of the Kalman fliter
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
    %[response,R] = make_response_matrix(num_cells,num_bins,experiment,cell_list,bin_size,start_time,0);
    response = experiment.spikes{1}(cell_list(:,1),:);
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
    
    % Initialise all the necesarry arrays for the prediction
    x_predictions = zeros(num_bins,1);
    y_predictions = zeros(num_bins,1);
    window_times = zeros(num_bins,1);
    
    x_predictions(1) = x;
    y_predictions(1) = y;
    
    % Initial values for state variables
    P_m = zeros(4,4,num_bins);
    P = zeros(4,4,num_bins);
    K = zeros(4,num_cells,num_bins);
    
    % Get the trained matricies
    A = filter.A;
    W = filter.W;
    H = filter.H;
    Q = filter.Q;

    % Initialise the step to two because (step-1) will be used
    step = 2;
      
    cur_time = start_time;
    window_times(1) = start_time + filter.lag*bin_size;
    
    % Preform the prediction based on Kalman algorithm
    while(step <= num_bins)

        % prior estimation 
        P_m(:,:,step) = A*P(:,:,step-1)*A'+W; 
        state_m(:,step) = A*state(:,step-1); 

        z = response(:,step);

        % posterior estimation
        K(:,:,step) = P_m(:,:,step)*H'*inv(H*P_m(:,:,step)*H'+Q);       
        P(:,:,step) = (eye(4)-K(:,:,step)*H)*P_m(:,:,step);   
        state(:,step) = state_m(:,step)+K(:,:,step)*(z-H*state_m(:,step));  

        cur_time = cur_time + bin_size;
        window_times(step) = cur_time;

        % Here we account for the fact that the filter was built to imply some lag between kinematics
        % and spikes...
        window_times(step) = window_times(step) + filter.lag*bin_size;

        x = state(1,step)+filter.center(1); % make X prediction
        y = state(2,step)+filter.center(2); % make Y prediction

        x_predictions(step) = x;
        y_predictions(step) = y;
        step = step + 1;
    end
    x_prediction = x_predictions(end);
    y_prediction = y_predictions(end);
end

% Format the testing data to fit our Kalman filter
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
