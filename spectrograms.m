% Create spectrograms from spike data
% Linnea Evanson
% 22/04/2021
clc; close all; clear all;

data = load("monkeydata_training.mat");

window = 10;
overlap = 5;

iter = 1;
for row = 1:100
    for trial = 1:8
        
        spikes = data.trial(row, trial).spikes;
        posn = data.trial(row, trial).handPos;
        
        timestep = 200;
        timechunk = 1;
        avg_data = sum(spikes(:,1:timestep)); % first 200 timesteps 
        nfft = 5*length(spikes);
        %spectrogram(avg_data, window, overlap, nfft, 'yaxis');
        spectro{row, trial, timechunk} = spectrogram(avg_data, window, overlap, nfft);
        
        prediction(row, trial, timechunk,:) = posn(1:2, timestep+1)  ; %x, y position in next time step
        
        stepsize = 20; % take data in 20 second chunks (as that's what we will get in test)
        for timestep = 201:stepsize:length(spikes) - stepsize -1
            timechunk = timechunk + 1;
            
            avg_data = mean(spikes(:,timestep:timestep+stepsize)); % first 200 timesteps 
            spectro{row, trial, timechunk} = spectrogram(avg_data);

            prediction(row, trial, timechunk,:) = posn(1:2, timestep+stepsize+1)  ; %x, y position in next time step

        end
        
    end
    iter = iter +1;
    %disp(string(iter*100/(8*100))+"% complete")
end

%Could make spectrogram images for each timestep where we get new data, and
%classify them with coordinates (continuous prediction).
