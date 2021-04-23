% Create spectrograms from spike data
% Linnea Evanson
% 22/04/2021
clc; close all; clear all;

%Make spectrogram images for each timestep where we get new data, and
%classify them with coordinates (continuous prediction).

%% Create dataset (images, that are spectrograms, created with certain window and overlap)

data = load("monkeydata_training.mat");

path = "spectro_images"; %where spectrograms are saved to

window = 10;
overlap = 5;
nfft = 600;

iter = 1;
for row = 1:100
    for trial = 1:8
        
        spikes = data.trial(row, trial).spikes;
        posn = data.trial(row, trial).handPos;
        
        %Pad the spikes so they are all the same length
%         l = length(spikes);
%         diff = 800 - l;
%         spikes = [zeros(98, floor(diff/2)), spikes, zeros(98, ceil(diff/2))];
        
%         
%         timestep = 200;
%         timechunk = 1;
%         avg_data = sum(spikes(:,1:timestep)); % first 200 timesteps 
%         nfft = 5*length(spikes);
%         %spectrogram(avg_data, window, overlap, nfft, 'yaxis');
%         spectro{row, trial, timechunk} = spectrogram(avg_data, window, overlap, nfft);
%         
%         prediction(row, trial, timechunk,:) = posn(1:2, timestep+1)  ; %x, y position in next time step
%         
        timechunk = 1;
        
        % We want all output spectrograms to be the same size, so we choose
        % step size to always divide into 30 chunks.
        num_chunks = 19;
        stepsize = round(length(spikes)/num_chunks);
        
        % For Nx length signal:
         %cols = fix((NX-NOVERLAP)/(length(WINDOW)-NOVERLAP)) 
        % ncols = 5;
        % window = fix((length(spikes) - overlap)/ncols + overlap);
        
        %stepsize = 20; % take data in 20 second chunks (as that's what we will get in test)

        for timestep = 1:stepsize:length(spikes) - stepsize -1
            
            avg_data = sum(spikes(:,timestep:timestep+stepsize),1); % first 200 timesteps 
            all_avg{row, trial, timechunk} = avg_data;
            if isempty(avg_data)
                disp("0!!!!!!!")
            end
            f = figure('visible','off'); %so spectrograms do not display
            spectrogram(avg_data, window, overlap, nfft);
            
            filename = string(row) + "_" + string(trial) + "_" + string(timestep) + ".png";
            saveas(gcf, fullfile( path, filename));
            close(gcf);
            
            %s = spectrogram(avg_data, window, overlap, nfft);
            
            spectro{row, trial, timechunk} = imread(fullfile( path, filename)); %, [301 4]);

            prediction(row, trial, timechunk,:) = posn(1:2, timestep+stepsize+1)  ; %x, y position in next time step
           
            timechunk = timechunk + 1

        end
        
    end
    iter = iter +1;
    %disp(string(iter*100/(8*100))+"% complete")
end
spectro = spectro(:,:,1:end-1);
prediction = prediction(:,:,1:end-1,:);

%The images are saved into one big spectro matrix, contains the spectrogram
%of each trial at each chunk of time (200 sec at start, then every 20 seconds after).

%% CNN with regression layer

% Inputs to neural net:
XTrain = cell2mat(spectro(1:50,:,:));
YTrain = cell2mat(prediction(1:50,:,:,:));

XValidation = cell2mat(spectro(51:100,:,:));
YValidation = cell2mat(prediction(51:100,:,:,:));

% Define network 
layers = [
    imageInputLayer([1900 39])   %image size   [28 28 1]
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
    fullyConnectedLayer(1)
    regressionLayer];

% Train network

miniBatchSize  = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',30, ...
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