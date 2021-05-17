%Pre competition work
%Linnea Evanson
%10/02/21
clc;
clear all;
close all;

data = load('monkeydata0.mat');
trials = data.trial;
trial1 = trials(1,1).handPos;
trial2 = trials(2,1).handPos;

figure()
plot(trial1(1,:))
hold on 
plot(trial1(2,:))
title("Trial 1")
legend("X data", "Y data")

figure()
plot(trial2(1,:))
hold on 
plot(trial2(2,:))
title("Trial 2")
legend("X data", "Y data")

%% Raster plots
%In a raster plot each row (y-axis) corresponds to the index of a neuron in
%a neuron group. The columns (x-axis) corresponds to the current time in the simulation.
%time should be in bins.

% For 1 trial
trial1spikes = trials(1,1).spikes; %size 98x672

x = [];
y = [];
for neuron = 1:98
    for timesteps = 1:672
        if trial1spikes(neuron,timesteps) == 1
            x = [x;neuron];
            y = [y;timesteps];
        end
    end
end
figure()
scatter(x,y,5,'filled'); % int indicates size of dots
xlabel("Time (ms)")
ylabel("Neuron Index")
title("Raster Plot of Trial 1");

%Raster plot for one neural unit over many trials
%different colour for each 8 trials
x = {};
y = {};
for trials = 1:8
    x_temp = [];
    y_temp = [];
    for movements = 1:8
        trial = data.trial(trials,movements);
        disp("number of timesteps:")
        disp(length(trial.spikes(1,:)))
        for timesteps = 1:length(trial.spikes(1,:))
            if trial.spikes(1,timesteps) == 1
                x_temp = [x_temp;timesteps];
                y_temp = [y_temp;movements];
            end
        end
        disp(x_temp)
    x{trials} = x_temp;
    y{trials} = y_temp;
    end
end

colors = ['r','g','b','y','m','c',[.8 .2 .6],'k']
figure()
for trials = 1:8
    scatter(x{1,trials}, y{1,trials}, 5, colors(trials),'filled') %each trial is a different colour
    hold on 
end
xlabel("Time (ms)")
ylabel("Action number")
ylim([0,8]);
legend("Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5", "Trial 6", "Trial 7", "Trial 8")
title("Raster Plot for Neuron 1")

%% Peri-stimulus time histograms
%Histogram of the times at which a neuron fires. One histogram per neural
%unit.

%Get times for all the trials, for Neuron one, for Task 1:
spikes = []
for row = 1:100
    trial = data.trial(row,1).spikes;
    disp(length(trial(1,:)))
    for timestep = 1:length(trial(1,:))
        neuron = trial(1,timestep);
        if neuron == 1
            spikes = [spikes;timestep];
        end
    end
end

windowWidth = 15; % Whatever you want.
kernel = ones(1,windowWidth) / windowWidth;

figure()
h = histogram(spikes,100)
hold on 
middle = [];
for i = 1:h.NumBins
    middle = [middle;(h.BinEdges(i) + h.BinEdges(i+1))/2  - windowWidth]
end
smooth = filter(kernel,1,h.Values); %smooth these to plot a smooth function of firing rate of the neuron
plot(middle, smooth, 'LineWidth',3)
xlabel("Time (ms)")
ylabel("Action potentials in 100 Trials")
legend("Action Potential Histogram", "Moving Average Firing Rate")
title("Peristimulus Time Histogram for Neuron 1, Task 1")








