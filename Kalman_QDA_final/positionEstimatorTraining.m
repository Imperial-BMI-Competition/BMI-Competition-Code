%%% Team Members: Joe Arrowsmith, Manfredi Castelli, Linnea Evanson, 
%%%               Joonsu Gha, Jon Skerlj
%%% BMI Spring 2021 (Update May 2021)

%%% Reference:
% Our implementation of the Kalman filter for neural decoding is based on
% that of Wu et al 2003 (https://papers.nips.cc/paper/2178-neural-decoding-of-cursor-motion-using-a-kalman-filter.pdf)
% The original implementation has previously been coded in Matlab by
% Dan Morris (http://dmorris.net/projects/neural_decoding.html#code)
 

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         PLEASE READ BELOW            %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function positionEstimator has to return the x and y coordinates of the
% monkey's hand position for each trial using only data up to that moment
% in time.
% You are free to use the whole trials for training the classifier.

% To evaluate performance we require from you two functions:

% A training function named "positionEstimatorTraining" which takes as
% input the entire (not subsampled) training data set and which returns a
% structure containing the parameters for the positionEstimator function:
% function modelParameters = positionEstimatorTraining(training_data)
% A predictor named "positionEstimator" which takes as input the data
% starting at 1ms and UP TO the timepoint at which you are asked to
% decode the hand position and the model parameters given by your training
% function:

% function [x y] = postitionEstimator(test_data, modelParameters)
% This function will be called iteratively starting with the neuronal data 
% going from 1 to 320 ms, then up to 340ms, 360ms, etc. until 100ms before 
% the end of trial.


% Place the positionEstimator.m and positionEstimatorTraining.m into a
% folder that is named with your official team name.

% Make sure that the output contains only the x and y coordinates of the
% monkey's hand.


function [modelParameters] = positionEstimatorTraining(training_data)
  
    % Classification part of the training
    % Uses QDA algorithm to classify the target of interest from the first
    % 320 ms of the neural data
    tr_s = size(training_data,1);
    
    idxes = randperm(tr_s, int8(tr_s * 1.0));
    tr_x = training_data(idxes,:);
    Pop_Vecs = SDA_decoder(tr_x);
    
    % Store the trained parameters for position estimation
    modelParameters.Pop_Vec = Pop_Vecs;
    modelParameters.retrain_sda = false;
    modelParameters.w = 0.8;
   
    
    bin_size = 0.01;
    % Set a threshold for minimum number of spikes per neuron for training
    min_spikes_per_cel = 400;
    % Set lag
    lag = 0;
    
    % Fit kalman filter from the training data
    modelParameters.kalmanFit = kalman_fit(training_data, bin_size, lag, min_spikes_per_cel);
    modelParameters.avg_endPos = avg_endPos(training_data);
end

% Function that orders the data into the correct format for training
function [formatted] = data_formatting(td)
    % get the size of the training data
    [N M] = size(td);
    % loop through the training data and store kinematics and neural spikes
    % separately for each of the 8 directions
    for m = 1:M
        samp_count = 0;
        for n = 1:N
            for t = 20:20:length(td(n,m).spikes)
                samp_count = samp_count + 1;
                x_pos{m}(samp_count,1) = 0.01*samp_count;
                x_pos{m}(samp_count,2) = td(n,m).handPos(1,t);
                y_pos{m}(samp_count,1) = 0.01*samp_count;
                y_pos{m}(samp_count,2) = td(n,m).handPos(2,t);
                
                spikes{m}(:,samp_count) = sum(td(n,m).spikes(:,t-19:t),2);
            end
        end
        % Count all the spikes for each neuron
        spike_count{m}=sum(spikes{m},2);
    end
    % Store the formatted data into a struct
    formatted.xpos = x_pos;
    formatted.ypos = y_pos;
    formatted.spikes = spikes;
    formatted.spike_count = spike_count;
end

function [filter] = kalman_fit(td,bin_size, lag, minimum_spikes_per_cell)
    % Format data
    experiment = data_formatting(td);
    % Train the Kalman filter for each of the 8th directions
    for target_id = 1:8
     
        % find cells with low spiking - they are going 
        cell_list = find_low_spiking_cells(experiment,minimum_spikes_per_cell, target_id);
        
        % get the final time
        end_time_idx = size(experiment.spikes{target_id}, 2);
        start_time = 0;
        % get the final index
        end_time = end_time_idx*bin_size;
        
        % store the different variables
        filter{target_id}.cell_list = cell_list;
        filter{target_id}.bin_size = bin_size;
        filter{target_id}.lag = lag;
        % Start and end time of training in seconds
        filter{target_id}.start_time = start_time;
        filter{target_id}.end_time = end_time;

        % compute the total number of bins in the specified time window
        num_bins = floor( (end_time - start_time) / bin_size);

        num_windows = num_bins;

        num_cells = size(cell_list,1);

        % Spikes matrix is 1-indexed, cell_list is 0-indexed
        % preform the correction
        cell_list = cell_list + 1;

        % Data used for training
        response = experiment.spikes{target_id}(cell_list(:,1),1:floor((end_time-start_time)/bin_size));

        % Remove last bins if there is some delay between spikes and
        % kinematics
        if (lag > 0)
            response = response(:,1:end-lag);
        end

        filter_length = 1;


        % create the stimulus vectors (binned kinematic data) for x and y

        s_x = zeros(num_windows,1);
        s_y = s_x;

        cur_x_index = 1;
        cur_y_index = 1;

        x_positions = experiment.xpos{target_id};
        y_positions = experiment.ypos{target_id};

        cur_x = -1;
        cur_y = -1;

        for(i=1:num_windows)
            window_start_time = start_time + (i-1)*bin_size;
            window_end_time = window_start_time + bin_size*filter_length;
            while(cur_x_index <= size(x_positions,1) & x_positions(cur_x_index,1) < window_end_time)
                cur_x = x_positions(cur_x_index,2);
                cur_x_index = cur_x_index + 1;
            end
            while(cur_y_index <= size(y_positions,1) & y_positions(cur_y_index,1) < window_end_time)
                cur_y = y_positions(cur_y_index,2);
                cur_y_index = cur_y_index + 1;
            end
            s_x(i) = cur_x;
            s_y(i) = cur_y;
        end

        % Remove first "lag" bins
        if (lag > 0)
            s_x = s_x(lag+1:end);
            s_y = s_y(lag+1:end);
        end
        % From here on, num_windows and num_bins will represent the number of bins actually stored
        % in the filter, not the number of bins present in the whole time period
        num_bins = num_bins - lag;
        num_windows = num_bins;

        % Store the stimulus and response matrices used to create this filter
        filter{target_id}.s_x = s_x;
        filter{target_id}.s_y = s_y;
        filter{target_id}.response = response;

        % Perform Kalman algorithm
        kinematics = [s_x s_y];

        % Compute the mean x and y values, since the filter operates on normalized
        % coordinates
        center = mean(kinematics(:,1:2));
        filter{target_id}.center = center;

        % Now normalize (subtract out the means)
        kinematics(:,1:2) = kinematics(:,1:2)-ones(num_windows,1)*center;

        % The experiment file provides only position; let's compute velocity
        kinematics(:,3:4) = zeros(num_windows,2);
        kinematics(2:num_windows,3:4) = diff(kinematics(:,1:2))/bin_size;

        % The kinematics array now contains one row for each time step, with each
        % row of the form :
        % [x_pos y_pos x_vel y_vel]

        % Compute the variables required for filtering, according to Wu et al
        X2 = kinematics(2:num_windows,:)';
        X1 = kinematics(1:num_windows-1,:)';

        % Compute the necessary matrices for the 'system model' ;
        % 
        % x_{k+1} = A*x_k+w_k

        % The least-squares-optimal transformation from x_i to x_(i+1)
        A = X2*X1'*inv(X1*X1');

        % The matrix of noise values for kinematics
        W = ((X2 - A*X1)*((X2 - A*X1)')) / (num_windows - 1);

        % A and W are both 4x4

        % Compute the necessary matrices for the 'generative model' of neural firing :
        %
        % z_k = H*x_k+q_k

        X = kinematics';
        Z = response;

        % The least-squares-optimal transformation from x_i to z_i
        % (the transformation from position to spikes)
        H = Z*X'*(inv(X*X'));

        % The covariance matrix for spikes
        Q = ((Z - H*X)*((Z - H*X)')) / num_windows;

        filter{target_id}.Q = Q;
        filter{target_id}.H = H;
        filter{target_id}.A = A;
        filter{target_id}.W = W;
    end

end

% Find indecies of neurons with number of spikes lower than the input 
% variable MIN_SPIKES
function [cells] = find_low_spiking_cells(experiment,MIN_SPIKES, target_id)

    s = experiment.spike_count{target_id};

    [i,j] = find(s > MIN_SPIKES);
    cells = [i j];

    % Return the 0-indexed numbering of the channels and units
    cells = cells - 1;

end


% Function that calculates the average ending position of every target
% of interest - used for restricting overshoots
function[end_pos] = avg_endPos(training_data)
    N = size(training_data,1);
    for i = 1:8
        grouped_end_pos = [[]];
        for j = 1:N
            handPos = training_data(j,i).handPos(1:2,end-100);
            grouped_end_pos = cat(3, grouped_end_pos, handPos);
        end
        end_pos(i).average = mean(grouped_end_pos,3); 
    end
end

% Function that calculates the average stating position of every target
% of interest
function [start_pos] = avg_start_pos(training_data)
    N = size(training_data,1);
    for i = 1:8
        grouped_start_pos = [[]];
        for j = 1:N
            handPos = training_data(j,i).handPos(1:2,250);
            grouped_start_pos = cat(3, grouped_start_pos, handPos);
        end
        start_pos(i).average = mean(grouped_start_pos,3); 
    end
end
