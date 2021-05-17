%% Linnea Evanson
% 05/05/2021
% Use LSTM network to predict precise coordinates of end of trajectory, to
% see whether it could work well for regression.

clc; clear all; close all;

load monkeydata_training.mat;

%Stack data from all trials:
trials = trial;
size_t = length(trials); %number of trials
dataTrain = cell(1,size_t*8); %needs to be in cell array for subsequent code

angles = [30*pi/180, 70*pi/180, 110*pi/180, 150*pi/180, 190*pi/180, 230*pi/180, 310*pi/180, 350*pi/180];

tic
count = 1;
for i = 1:size_t %number of trials
    for j = 1:8     %number of types of directions (experiments)
        
        %Don't train on first 320 ms because they have unpredictable
        %trajectories
%         time = 1;
%         %N = trials(i,j).handPos(1:2,time:320); %select only x and y, not z
%         N = zeros(2,320); %first hand movements don't matter, don't try to predict them
%         S = trials(i,j).spikes(:,time:320);
%         
%         datax{count} = S;
%         datay{count} = N;
%         
%         count = count+1;
        
        for t = 320:20:size(trials(i,j).spikes,2)-20
            N = trials(i,j).handPos(1:2,t:t+20); %select only x and y, not z
            S = trials(i,j).spikes(:,t:t+20);

            datax{count} = S;
            datay{count} = N;

            count = count+1;
            
        end
        
    end
end

split = 0.8;
XTrain = datax(:,1: round(split*length(datax)));
YTrain = datay(:, 1: round(split*length(datay)));

XValidation = datax(:, round(split*length(datax)+1:end));
YValidation = datay(:, round(split*length(datay)+1:end));


% Initialise network
% From https://www.mathworks.com/help/deeplearning/ug/long-short-term-memory-networks.html
numHiddenUnits = 100;
numResponses = size(YTrain{1},1);
numFeatures = size(XTrain{1},1); %x, y and z. later we will change this to be more than just hand position


layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence'); % 'OutputMode','last') ; % %for a single prediction output (not a sequence)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% From https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html
minibatchsize = 100;
validationFrequency = floor(numel(YTrain)/minibatchsize);
options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize',minibatchsize, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
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

toc
