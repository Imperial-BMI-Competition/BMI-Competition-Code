%% Linnea Evanson
% 01/05/2021
% LSTM neural decoder to classify neural data based on first 320 seconds.
clc; clear all; close all;

load monkeydata_training.mat;
 
%Stack data from all trials:
trials = trial;
size_t = length(trials); %number of trials
dataTrain = cell(1,size_t*8); %needs to be in cell array for subsequent code

%angles = [30*pi/180, 70*pi/180, 110*pi/180, 150*pi/180, 190*pi/180, 230*pi/180, 310*pi/180, 350*pi/180];

time_range = 320;

tic 
count = 1;
for i = 1:size_t %number of trials
    for j = 1:8     %number of types of directions (experiments)
        
        N = trials(i,j).handPos(1:2,1:time_range); %select only x and y, not z
        end_pos = N(:,end); %final position
        
        S = trials(i,j).spikes(:,1:time_range);
        n = size(N);

        datax{count} = S;
        datay{count} = j;
        
        count = count+1;
    end
end


% Initialise network
% From https://www.mathworks.com/help/deeplearning/ug/long-short-term-memory-networks.html
numHiddenUnits = 100;
numResponses = 8 ;%size(YTrain{1},1);
numFeatures = size(datax{1},1); %x, y and z. later we will change this to be more than just hand position

XTrain = datax(:,1:0.8*800)';
YTrain = categorical(cell2mat(datay(:,1:0.8*800))');

XValidation = datax(:,0.8*800 + 1:end)';
YValidation = categorical(cell2mat(datay(:,0.8*800+1 : end))');

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','last') ; %'OutputMode','sequence');  %for a single prediction output (not a sequence)
    fullyConnectedLayer(numResponses)
    softmaxLayer
    classificationLayer];

miniBatchSize = 50;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
% From https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html
options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
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

%% Validation 
num_trials = 8;
preds = classify(net, XValidation);
preds = categorical(preds);

train_preds = classify(net, XTrain);
train_preds = categorical(train_preds);

train_acc = 100*sum(train_preds == YTrain)/length(train_preds);
disp("Final Training Accuracy: " + train_acc + "%")

acc = 100*sum(preds == YValidation)/length(preds);
ACo = 1/num_trials;
disp("Final Validation Accuracy: " + acc + "%")
disp("Kappa coefficient: " + (acc/100 - ACo)/(1-ACo))

figure
histogram(double(preds))
hold on; histogram(double(YValidation))
title("Trial Predictions")
legend(["Predicted" "True"])

C = confusionmat(double(preds), double(YValidation));
figure()
confusionchart(double(preds), double(YValidation));
title("Confusion Matrix of Trials using LSTM")
