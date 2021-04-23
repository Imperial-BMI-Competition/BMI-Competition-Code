% Script to train and test LSTM model
% Linnea Evanson
% 27/02/2021
clc;
close all;
clear all;

%Load data, select one trial for playing around with:
all_data = load('monkeydata_training.mat')
trials = all_data.trial;
pos = trials(1,1).handPos; 
data = pos;

% Split into train and test, and normalise
size_data = size(data);
numTimeStepsTrain = floor(0.9*size_data(2));

dataTrain = data(:, 1:numTimeStepsTrain+1);
dataTest = data(:, numTimeStepsTrain+1:end);

mu = mean(dataTrain,2);
sig = std(dataTrain,0,2); %0 weights, dimension 2 (rows)
dataTrainStandardized = (dataTrain - mu) ./ sig;

XTrain = dataTrainStandardized(:,1:end-1);  %to predict use all data but end time step
YTrain = dataTrainStandardized(1:2,2:end); %!!!!!!!!!!!!! choose only x and y coordinates to predict

%Normalise test set
dataTestStandardized = (dataTest - mu) ./ sig;
XTest = dataTestStandardized(:,1:end-1);

%Now convert these to cells
XTrain = num2cell(XTrain,1);
YTrain = num2cell(YTrain,1);
XTest = num2cell(XTest,1);


%Initialise network
% From https://www.mathworks.com/help/deeplearning/ug/long-short-term-memory-networks.html
numHiddenUnits = 500;
numResponses = size(YTrain{1},1);
numFeatures = size(XTrain{1},1); %x, y and z. later we will change this to be more than just hand position


layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence') %,'OutputMode','last') %for a single prediction output (not a sequence)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% From https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html
options = trainingOptions('adam', ...
    'MaxEpochs',5, ...
    'MiniBatchSize',1, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...  %after 125 epochs drop learning rate by factor of 0.2
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

%Train network:
net = predictAndUpdateState(net,XTrain, 'MiniBatchSize', 1);
[net,YPred] = predictAndUpdateState(net,YTrain(end), 'MiniBatchSize', 1);

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu', 'MiniBatchSize', 1);
end

%Unnormalise and find RMSE of prediction:
YPred = cell2mat(YPred)
YPred = sig.*YPred + mu;

YTest = dataTest(:,2:end);
rmse = sqrt(mean((YPred-YTest).^2))

%Compare forcast with ground truth:
figure
subplot(2,1,1)
plot(YTest(1,:))
hold on
plot(YPred(1,:),'.-')
hold on
plot(YTest(2,:))
hold on
plot(YPred(2,:),'.-')
hold on
plot(YTest(3,:))
hold on
plot(YPred(3,:),'.-')
hold on
legend(["Observed X" "Forecast X" "Observed Y" "Forecast Y" "Observed Z" "Forecast Z"])
ylabel("Magnitude")
xlabel("Timestep")
title("Predicted Coordinates")

subplot(2,1,2)
stem(YPred(1,:) - YTest(1,:))
hold on
stem(YPred(2,:) - YTest(2,:))
hold on
stem(YPred(3,:) - YTest(3,:))
legend(["X" "Y" "Z"])
xlabel("Month")
ylabel("Error")
title("RMSE")


figure
plot(YPred(1,:), YPred(2,:))
hold on
plot(YTest(1,:), YTest(2,:))
legend(["Forecast position" "Observed Position"])
xlabel("X Coordinate")
ylabel("Y Coordinate")
title("Predicted and Observed Position")


disp("RMSE")
rmse

