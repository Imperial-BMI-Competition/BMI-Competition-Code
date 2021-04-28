% Use trained network to predict:

% 
% YPredicted = predict(net,XValidation);
% 
% predictionError = YValidation - YPredicted;
% 
% squares = predictionError.^2;
% rmse = sqrt(mean(squares))

clc; clear all; close all;

data = load("monkeydata_training.mat");

prediction = zeros(8* 100,2);
iter = 1;
tic
for row = 1:100
    for trial = 1:8
        
        posn = data.trial(row, trial).handPos;
        
        timechunk = 1;
        
        % We want all output spectrograms to be the same size, so we choose
        % step size to always divide into 30 chunks.
        num_chunks = 1;
        stepsize = round(200/num_chunks);
        
        % For Nx length signal:
         %cols = fix((NX-NOVERLAP)/(length(WINDOW)-NOVERLAP)) 
        % ncols = 5;
        % window = fix((length(spikes) - overlap)/ncols + overlap);
        
        %stepsize = 20; % take data in 20 second chunks (as that's what we will get in test)

        for timestep = 1 %:stepsize:200 - stepsize -1   %just do first 200 timesteps only
            
            prediction(iter,:) = posn(1:2, end); %x, y position in next time step
           
            timechunk = timechunk + 1;

        end
    iter = iter +1;

    end
    disp(string(iter*100/(8*100))+"% complete")
end
toc

images = load("spectro_1.mat");
spectro = images.spectro;
%% CNN with regression layer

% Inputs to neural net:
XTrain = spectro(:,:,:,1:400); %cell2mat(spectro(1:50,:,:));
YTrain = prediction(1:400,:);

XValidation = spectro(:,:,:,401:end); %cell2mat(spectro(51:100,:,:));
YValidation = prediction(401:end,:); %cell2mat(prediction(51:100,:,:,:));


% Define network 
layers = [
    imageInputLayer([429 543 3])   %image size   [28 28 1] [429 543 3]
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling2dLayer(2,'Stride',2)
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(2)  %changed output size to 2, because x and y component
    regressionLayer];

% Train network

miniBatchSize  = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...   % learning rate drops after 20 epochs
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);


net = trainNetwork(XTrain,YTrain,layers,options);