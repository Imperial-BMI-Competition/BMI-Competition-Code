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
    model_lstm = LSTM_training(training_data);
    
    decoder = angle_decoder_training(data);
    
    modelParameters{1,1} = model_lstm; 
    modelParameters{1,2} = decoder; 
    

end



function [model_minibatch16_neural] = LSTM_training(trials)

    size_t = length(trials); %number of trials
    dataTrain = cell(1,size_t*8); %needs to be in cell array for subsequent code


    count = 1;
    for i = 1:size_t %number of trials
        for j = 1:8     %number of types of directions (experiments)
            N = trials(i,j).handPos(1:2,:); %select only x and y, not z
            S = trials(i,j).spikes;
            dataTrain{count} = [N;S]; % make one long row vector with features of position and spikes
            count = count+1;
        end
    end

    %Standardise all the features by mean of that feature across all trials
    dataTrain2 = cell2mat(dataTrain);  %to calculate mean and standard deviation of each feature to normalise
    train_mu = mean(dataTrain2,2);
    train_sig = std(dataTrain2,0,2); %0 weights, dimension 2 (rows)

    dataTrain_standardisedX = cell(1,size_t*8);
    dataTrain_standardisedY = cell(1,size_t*8);

    for i = 1:size_t*8
        trial = dataTrain{i};
        trial = (trial - train_mu) ./ train_sig;
        dataTrain_standardisedX{i} = trial(:,1:end-1);
        dataTrain_standardisedY{i} = trial(:,2:end);
    end

    XTrain = dataTrain_standardisedX;  %to predict use all data but end time step
    YTrain = dataTrain_standardisedY; 

    disp("Created test dataset")


    % Initialise network
    % From https://www.mathworks.com/help/deeplearning/ug/long-short-term-memory-networks.html
    numHiddenUnits = 100;
    numResponses = size(YTrain{1},1);
    numFeatures = size(XTrain{1},1); %x, y and z. later we will change this to be more than just hand position


    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits,'OutputMode','sequence') %,'OutputMode','last') %for a single prediction output (not a sequence)
        fullyConnectedLayer(numResponses)
        regressionLayer];

    % From https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html
    options = trainingOptions('adam', ...
        'MaxEpochs',2, ...
        'MiniBatchSize',16, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...  %after 125 epochs drop learning rate by factor of 0.2
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress');

    disp("Initialised Network")

    % Train Network
    net = trainNetwork(XTrain,YTrain,layers,options);

    % Test Network:
    net = predictAndUpdateState(net,XTrain, 'MiniBatchSize', 1);  %initialise network

    modelParameters.net = net;
    modelParameters.mu = train_mu;
    modelParameters.sig = train_sig;

    model_minibatch16_neural = modelParameters;
    %save model_minibatch16_lr0.01_epoch10_neural
end

function [decoder] = angle_decoder_training(data)
    % Estiamtes Neurons Tuning curve for population vector decoding

    fsamp = 1000;
    angles = [30    70   110   150   190   230  ,   310   350];

    %Get the dishcarge rates of all  neurons, angles and tasks
    for neuron  = 1: size(data.trial(1,1).spikes,1)
        for angle = 1:size(data.trial,2)
            spikes = [];
            spikes_ =  [];
            for row = 1:100
                T = size(data.trial(row,angle).spikes,2);
                trial = data.trial(row,angle).spikes;
                times = sum(trial(neuron,1:T));
                dr = times ./ (T/fsamp) ;
                spikes = [spikes,dr];

                times_ = sum(trial(neuron,1:200));
                dr_ = times_ ./ (200/fsamp) ;
                spikes_ = [spikes_,dr];
                spike = find(trial(neuron,1:end)==1);
                pps = fsamp./diff(spike);
  
            end
            test_rates(neuron,angle,:) = spikes_; 
            all_rates(neuron,angle,:) = spikes; 
            firing_rate(neuron,angle) = nanmean(spikes,2);
            error = nanstd(spikes,[],1);

        end
    end
    
    % Check of neurons are directionally tuned 
    theta_radians = deg2rad(angles);
    [x, y] = pol2cart(theta_radians, 1);  
    unit_vectors = [x;y];

    [r_max,s_a] = max(firing_rate,[],2);
    directional_tuning = [];
    % r_max = (r_max - mean(firing_rate, 2)); % If spike much bigger than mean-> good tuning neuron-> bigger weight
    C_neur = [];
    for neuron = 1 : size(data.trial(1,1).spikes,1)
        pref_dir = theta_radians(s_a(neuron));
        [x, y] = pol2cart(pref_dir, 1);  
        C_neur = [C_neur,[x;y]];
        directional_tuning(neuron) = nanstd(firing_rate(neuron,:),[],2);

        fa_s(neuron,:) = mean(firing_rate(neuron,:)) +  r_max(neuron) .* cos((theta_radians - pref_dir));

    end

    directional_threshold = 0.5;
    % directional_tuning(directional_tuning < directional_threshold) = [];

    n_dir_neurons = sum(directional_tuning < directional_threshold);
    fprintf('%i %% of neurons showed no directional tuning and were discarded.\n',round((n_dir_neurons/size(data.trial(1,1).spikes,1)) * 100));

    discard = true;
    non_directional = directional_tuning < directional_threshold;
    
    % If wish not to discard non-directional neurons
    if ~discard
        non_directional = ones(size(r_max));
    end
    
    
    decoder = {};
    
    % Format Output
    decoder{1,1} = 'Tuning_Curve';
    decoder{2,1} = 'Preferred_angle';
    decoder{3,1} = 'Non_direction_neurons';
    
    decoder{1,2} = fa_s;
    decoder{2,2} = C_neur;
    decoder{3,2} = non_directional;
       
end
