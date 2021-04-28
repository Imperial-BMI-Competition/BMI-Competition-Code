% Create spectrograms from spike data
% Linnea Evanson
% 22/04/2021
clc; close all; clear all;

%Make spectrogram images for each timestep where we get new data, and
%classify them with coordinates (continuous prediction).

%% Create dataset (images, that are spectrograms, created with certain window and overlap)

data = load("monkeydata_training.mat");

path = "spectro_images"; %where spectrograms are saved to

window = 100;
overlap = 0.5*window;
nfft = 600;
fs = 1;  %normalised sampling frequency

cmap = colormap;
iter = 1;

num_row = 100;
num_trials = 8;
num_timechunks = 1;
spectro = zeros(429, 543, 3, num_row* num_trials) ; %cell(num_row* num_trials,1);
prediction = zeros(num_row* num_trials,2);
tic
for row = 1:100
    for trial = 1:8
        
        spikes = data.trial(row, trial).spikes;
        posn = data.trial(row, trial).handPos;
        
        timechunk = 1;
        
        % We want all output spectrograms to be the same size, so we choose
        % step size to always divide into 30 chunks.
        num_chunks = 1;
        stepsize = round(200/num_chunks);
        
        for timestep = 1 %:stepsize:200 - stepsize -1   %just do first 200 timesteps only
            
            avg_data = sum(spikes(:,timestep:timestep+stepsize),1); 
            all_avg{row, trial, timechunk} = avg_data;
            if isempty(avg_data)
                disp("0!!!!!!!")
            end
            f = figure('visible','off');
            [~,f,t, psd] = spectrogram(avg_data, window, overlap, nfft, fs);
            s = surf(t,f,10*log10(abs(psd)),'edgecolor','none');
            view(2);
            set(gca, 'Visible', 'off');
            colorbar('off');
            F = getframe;
            A = frame2im(F);
            
            spectro(:,:,:,iter) =  A;
            prediction(iter,:) = posn(1:2, end); %x, y position in next time step
           
            timechunk = timechunk + 1;

        end
    iter = iter +1;

    end
    disp(string(iter*100/(8*100))+"% complete")
end
toc
disp(toc-tic)
%spectro = spectro(:,:,1:end-1);
%prediction = prediction(:,:,1:end-1,:);

%The images are saved into one big spectro matrix, contains the spectrogram
%of each trial at each chunk of time (200 sec at start, then every 20 seconds after).

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