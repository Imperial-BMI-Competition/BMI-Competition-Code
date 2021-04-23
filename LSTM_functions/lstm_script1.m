% Script to train and test LSTM model
% Linnea Evanson
% 27/02/2021
clc;
close all;
clear all;

%Load data:
all_data = load('monkeydata_training.mat');
trials = all_data.trial;

%Stack data from all trials:
dataTrain = cell(1,640);  %there are 800 trials,  80% train
dataTest = cell(1,160);   % 20% test.
count = 1;
for i = 1:80
    for j = 1:8
        N = trials(i,j).handPos;
        dataTrain{count} = N; % make one long row vector
        count = count+1;
    end
end

count = 1;
for i = 1:20
    for j = 1:8
        N = trials(i+80, j).handPos; %put remaining trials in test
        dataTest{count} = N; % make one long row vector
        count = count +1;
    end
end


%Standardise all the features by mean of that feature across all trials
dataTrain2 = cell2mat(dataTrain);  %to calculate mean and standard deviation of each feature to normalise
train_mu = mean(dataTrain2,2);
train_sig = std(dataTrain2,0,2); %0 weights, dimension 2 (rows)

dataTrain_standardisedX = cell(1,640);
dataTrain_standardisedY = cell(1,640);

for i = 1:640
    trial = dataTrain{i};
    trial = (trial - train_mu) ./ train_sig;
    dataTrain_standardisedX{i} = trial(:,1:end-1);
    dataTrain_standardisedY{i} = trial(:,2:end);
end

dataTest2 = cell2mat(dataTest);  %to calculate mean and standard deviation of each feature to normalise
test_mu = mean(dataTest2,2);
test_sig = std(dataTest2,0,2); %0 weights, dimension 2 (rows)

dataTest_standardised = cell(1,160);
YTest = cell(1,160); %this does need need to be standardised as isn't put through the network
for i = 1:160
    trial = dataTest{i};
    YTest{i} = trial(:,2:end);

    trial = (trial - test_mu) ./ test_sig;
    dataTest_standardised{i} = trial(:,1:end-1); %don't predict on very last timestep
    
end

XTrain = dataTrain_standardisedX;  %to predict use all data but end time step
YTrain = dataTrain_standardisedY; 

XTest = dataTest_standardised;

disp("Created test and train datasets")


%% Initialise network
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

%% Train Network
net = trainNetwork(XTrain,YTrain,layers,options);

%% Load Trained Network
% disp("Loading network")
% load trained_lstm

%% Test Network:
net = predictAndUpdateState(net,XTrain, 'MiniBatchSize', 16);  %initialise network

disp("Begin testing")
[net,YPred] = predictAndUpdateState(net,XTest,'ExecutionEnvironment','cpu', 'MiniBatchSize', 16)

%% Unnormalise predictions
YPred_units = cell(1,160);
meanSqError = 0;
for t = 1:160 % number of test trials
    y = cell2mat(YPred(t));
    y = test_sig.*y + test_mu;
    YPred_units{t} = y;    
end

%Find RMSE of prediction:
yp_all = cell2mat(YPred_units); %convert to arrays to find rmse
yt_all = cell2mat(YTest);
disp("RMSE")
%rmse = sqrt(mean((yp_all -yt_all).^2, 'all'))
rmse = sqrt( sum( (vecnorm(yp_all -yt_all)).^2 ) / numel(yp_all)  )  %vecnorm is norm along of each column in matrix.

%Plot all predicted and test trajectories
figure()
for i = 1:160  
    yp = YPred_units{i};
    yt = YTest{i};
    plot(yp(1,:), yp(2,:), 'r')
    hold on
    plot(yt(1,:), yt(2,:), 'b')
    hold on 
end
legend(["Predicted" "Ground Truth"])
title("Test Trajectories LSTM")
hold off


