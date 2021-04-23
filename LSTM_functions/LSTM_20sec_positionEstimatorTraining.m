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
  
  
%Stack data from all trials:
trials = training_data;
size_t = length(trials); %number of trials
dataTrain = cell(1,size_t*8); %needs to be in cell array for subsequent code


%Create features: we want data every 20 sec. Change in position, and sum of
%all neural spikes in that 20 sec window.
%We also want int which gives experiment type (angle of reach) (from 1-8).

count = 1;
for i = 1:size_t %number of trials
    for j = 1:8     %number of types of directions (experiments)
        
        N = trials(i,j).handPos(1:2,:); %select only x and y, not z
        S = trials(i,j).spikes;
        n = size(N);
        spikes20 = zeros(98,floor(n(2)/20));
        dpos20 = zeros(2,floor(n(2)/20)); %change in position after 20ms
        diff = N(:,1);
        
        k_count = 1;
        for k = 1:20:320 -20
            spikes20(:,k_count) = mean( S(:,k:k+19), 2); %sum everything we will get in very first time step
            dpos20(:,k_count) =  diff;  %N(:,1) - N(:,320);    %in test we won't know change for first 320 ms, so let's train that way
            diff = [0.0;0.0];
            k_count= k_count + 1;
        end
        
        initial = N(:,1);
        for k = 320:20:length(N)-20 %iterate through all other timesteps in 20 sec windows
            spikes20(:,k_count) = mean( S(:,k:k+19),2) ; %sum all spikes in 20 sec window
            dpos20(:,k_count) = initial - N(:,k+19);
            initial = N(:,k+19);
            k_count = k_count + 1;
        end
        experiment = ones(1,length(dpos20))*j;
        dataTrain{count} = [dpos20;experiment;spikes20]; % make one long row vector with features of position and spikes
        count = count+1;
    end
end

%Standardise all the features by mean of that feature across all trials
dataTrain2 = cell2mat(dataTrain);  %to calculate mean and standard deviation of each feature to normalise
% train_mu = mean(dataTrain2,2);
% train_sig = std(dataTrain2,0,2); %0 weights, dimension 2 (rows)

train_mu = 0;
train_sig = 1;

dataTrain_standardisedX = cell(1,size_t*8/20);
dataTrain_standardisedY = cell(1,size_t*8/20);

for i = 1:size_t*8/20
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
    'MaxEpochs',350, ...
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


