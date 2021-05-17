% Create spectrograms from spike data
% Linnea Evanson
% 22/04/2021
clc; close all; clear all;

%Make spectrogram images for each timestep where we get new data, and
%classify them. This is set to use just 10 trials, and take under 5
%minutes.

%% Create dataset (images, that are spectrograms, created with certain window and overlap)
tic
% data = load("monkeydata_training.mat");
% 
% path = "spectro_images"; %where spectrograms are saved to
% 
% window = 100;
% overlap = 0.5*window;
% nfft = 600; 
% fs = 1;  %normalised sampling frequency
% 
% cmap = colormap;
% iter = 1;
% 
%num_row = 10;
%num_trials = 2;
%trials = [2 5];
% trials = [1 2];
% num_trials = length(trials);
% num_timechunks = 1;
% spectro = zeros(32, 32, 98, 3, num_row* num_trials) ; %cell(num_row* num_trials,1);
% prediction = zeros(num_row* num_trials,1);
% 
% first_seconds = 320;
% 
% bottom = -50;
% top = 1;
% 
% 
% for row = 1:num_row
%     for i = 1:num_trials
%         trial = trials(i);
%         spikes = data.trial(row, trial).spikes;
%         posn = data.trial(row, trial).handPos;
%         
%         timechunk = 1;
%         
%         % We want all output spectrograms to be the same size, so we choose
%         % step size to always divide into 30 chunks.
%         num_chunks = 1;
%         stepsize = round(first_seconds/num_chunks);
%         
%         one_trial_spectro = zeros(32, 32, 98, 3);
%         
%         for timestep = 1 %:stepsize:200 - stepsize -1   %just do first 200 timesteps only
%             
%             
%             parfor neuron = 1:98
%                 one_neuron = spikes(neuron,timestep:timestep+stepsize);
%                 [~,freq,t, psd] = spectrogram(one_neuron, window, overlap, nfft, fs);
%                 f = figure('visible','off');
%                 s = surf(t,freq,10*log10(abs(psd)),'edgecolor','none');
%                 view(2);
%                 set(gca, 'Visible', 'off');
%                 colorbar('off');
% 
%                 caxis manual
%                 caxis([bottom top]);
%                 xlim([50 250]);
%                 ylim([0 0.5]);
%                 
%                 F = getframe;
%                 A = frame2im(F);
%                 close(f);
%                 one_trial_spectro(:,:,neuron,:) = imresize(A,[32 32]); %make a 3D image, where depth is the 98 neurons
%                 
%             end
%             
%             spectro(:,:,:, :, iter) =  one_trial_spectro;
% 
%             prediction(iter,:) =  trial; %posn(1:2, end); %x, y position in next time step 
%            
%             timechunk = timechunk + 1;
% 
%         end
%     iter = iter +1;
%     
%     end
%     disp(string(iter*100/(num_trials*num_row))+"% spectrograms computed")
%     
% end


%% Load dataset of precomputed spectrograms:

load("spectro_100perc_trial2and5.mat");
load("prediction_100perc_trial2and5_categorical.mat");

num_trials = 2;
num_row = 100;


%% CNN with regression layer

data_points = num_trials*num_row;   % 80/20 train/val split
train_points = round(0.8*data_points);

% Inputs to neural net:
XTrain = spectro(:,:,:,:,1:train_points); 
YTrain = categorical(prediction(1:train_points,1));

XValidation = spectro(:,:,:,:,train_points:end); 
YValidation = categorical(prediction(train_points:end,1)); 


%% Define network 
layers = [
    image3dInputLayer([32 32 98 3])   %image size 
    convolution3dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling3dLayer(2,'Stride',2)
    dropoutLayer(0.1)

    convolution3dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    averagePooling3dLayer(2,'Stride',2)
    dropoutLayer(0.1)
%     convolution3dLayer(3,32,'Padding','same')  % removed because it
%     overfit
%     batchNormalizationLayer
%     reluLayer
%     %dropoutLayer(0.2)
    convolution3dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(num_trials)  %changed output size to 2, because 2 trials
    softmaxLayer
    classificationLayer];

% Train network

miniBatchSize  = 50;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...   % learning rate drops after 20 epochs
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);


net = trainNetwork(XTrain,YTrain,layers,options);
toc


%% Predict and make confusion matrix:
preds = classify(net, XValidation);
preds = categorical(preds);

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
title("Confusion Matrix of Trials using Spectrogram Method")
