%%% Team Members: WRITE YOUR TEAM MEMBERS' NAMES HERE
%%% BMI Spring 2015 (Update 17th March 2015)

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
  % Arguments:
  
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model
  
  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.
  
  
    %Train LSTM Model For forecasting position:
%     decoder = angle_decoder_training(training_data);
    decoder = SDA_decoder(training_data);
    %model_lstm = LSTM_training(training_data);
    Velocity = average_velocities(training_data);
    startPos = avg_start_pos(training_data);
    %modelParameters.LSTM = model_lstm; 
    modelParameters.Pop_Vec = decoder;
    modelParameters.Vel = Velocity;
    modelParameters.Start_Pos = startPos;
    

end

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

function [velocity] = average_velocities(training_data)
    N = size(training_data,1);
    td = add_vel(training_data);
    % find average velocities
    for i = 1:8
    Grouped_coord = ([[]]);
    min = 500;
        for j = 1:N
            start = 300;
            stop = size(td(j,i).handVel,2);
           if (stop-start)<min
              min = stop - start; 
           end
        end
        for j = 1:N
           start = 300;
           stop = start + min;

           coord = td(j,i).handVel(:, start:stop);
           Grouped_coord = cat(3, Grouped_coord, coord);
        end
        velocity(i).average = mean(Grouped_coord,3);
    end
    
    
end
