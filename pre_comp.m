%Pre competition work
% Manfredi Castelli 
% 01/03/2021
clc;
clear all;
close all;

data = load('monkeydata_training.mat');
trials = data.trial;
trial1 = trials(1,1).handPos;
trial2 = trials(2,1).handPos;
%%
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

colors = jet(size(data.trial,1));
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
        disp("number of timesteps:");
        disp(length(trial.spikes(1,:)));
        for timesteps = 1:length(trial.spikes(1,:))
            if trial.spikes(1,timesteps) == 1
                x_temp = [x_temp;timesteps];
                y_temp = [y_temp;movements];
            end
        end
        disp(x_temp);
    x{trials} = x_temp;
    y{trials} = y_temp;
    end
end


figure()
for trials = 1:8
    scatter(x{1,trials}, y{1,trials},'MarkerFaceColor', colors(trials,:),'linewidth',1); %each trial is a different colour
    hold on ;
end
xlabel("Time (ms)");
ylabel("Action number");
ylim([0,8]);
legend("Trial 1", "Trial 2", "Trial 3", "Trial 4", "Trial 5", "Trial 6", "Trial 7", "Trial 8");
title("Raster Plot for Neuron 1");
%% PCA

figure()
n_pc = 8;
y_ticks = {};
cols = jet(n_pc);
for PC = 1:n_pc
plot(score(:,PC),PC*2,'o','color',cols(PC,:),'linewidth',1); %each trial is a different colour
y_ticks{PC} = sprintf('PC %i',PC);
hold on
end
xlabel("Time (ms)");
ylabel("Action number");
yticks(2.*[1:n_pc]);
yticklabels(y_ticks);
title("Raster Plot for Neuron 1");
%% Peri-stimulus time histograms
%Histogram of the times at which a neuron fires. One histogram per neural
%unit.
figure()
fsamp = 1000;
angle = 1;
neuron = 5;
%Get times for all the trials, for Neuron one, for Task 1:
for angle = 1:8
spikes = [];
cst = [];
% for row = 1:100
%     trial = data.trial(row,angle).spikes;
%     times = find(trial(neuron,:)== 1);
%     dr = fsamp./(diff(times));
%     spikes = [spikes,dr];
% end

for row = 1:100
    trial = data.trial(row,1).spikes;
    for timestep = 1:length(trial(1,:))
        neur = trial(1,timestep);
        if neur == 1
            spikes = [spikes;timestep];
        end
    end
end
% cst = sum(spikes,1)./fsamp;
windowWidth = 15; % Whatever you want.
kernel = ones(1,windowWidth) / windowWidth;


subplot(3,3,angle)
h = histogram(spikes,100);
hold on ;
middle = [];
for i = 1:h.NumBins
    middle = [middle;(h.BinEdges(i) + h.BinEdges(i+1))/2  - windowWidth];
end
smooth = filter(kernel,1,h.Values); %smooth these to plot a smooth function of firing rate of the neuron
plot(middle, smooth, 'LineWidth',3);
hold off;
ylim([0,40]);
xlabel("Time (msec)")
ylabel("Distribution")

title(sprintf("Peristimulus Time Histogram for Neuron %i, Angle %i", neuron,angle));

end
legend("Action Potential Histogram", "Moving Average Firing Rate")
%% Frequency domain analysis 
fsamp = 1000;
uu = 1;
for neuron  = 1: size(data.trial(1,1).spikes,1) - 1
    for next_neur = neuron : size(data.trial(1,1).spikes,1)
        for angle = 1
    %         for angle = 1:size(data.trial,2)
            density = [];
            for row = 1
                trial = data.trial(row,angle).spikes;
                first = trial(neuron,:);
                next = trial(next_neur,:);
    %             [psd,F_psd] = pwelch(trial(neuron,:), hann(0.2*fsamp), [], [], fsamp , 'psd');
                [psd,F_psd] = mscohere(normalize(first,'center'),normalize(next,'center'),...
            hann(0.2*fsamp),[],[],fsamp) ;
                density = [density; psd]; 
            end

            MSC_avg_trials(uu,:) = nanmean(density,2);
            uu = uu + 1;
    %         error_psd = nanstd(squeeze(all_psd(neuron,angle,:)),[],1);
    end
    end
end

%% %Visualize spectrum 
cols = jet(size(data.trial,2));
figure; 
for i = 1: size(data.trial,2)
    stdshade(squeeze(all_psd(:,i,:)),0.2,cols(i,:),F_psd); hold on;
%     plot(F_psd,squeeze(MSC_avg_trials(i,1,:)),'color',cols(1,:)); hold on;
end
xlabel('Frequency (Hz)','FontSize', 14, 'Color', 'k')
ylabel('PSD (AU)','FontSize', 14, 'Color', 'k')
xlim([0,200]);
%% TUning 
%% Peri-stimulus time histograms
%Histogram of the times at which a neuron fires. One histogram per neural
%unit.

fsamp = 1000;
angle = 1;
neuron = 5;
inst_rates = {};
%Get times for all the trials, for Neuron one, for Task 1:

for neuron  = 1: size(data.trial(1,1).spikes,1)
    for angle = 1:size(data.trial,2)
        spikes = [];
        for row = 1:100
            trial = data.trial(row,angle).spikes;
            times = sum(trial(neuron,1:200));
            dr = times ./ (size(trial,2)/fsamp) ;
            spikes = [spikes,dr];
            
            
            spike = find(trial(neuron,1:200)==1);
            pps = fsamp./diff(spike);
%             inst_rates{neuron,angle,row} = trial(neuron,1:200) ; 
%             inst_rates{neuron,angle,row,spike(2:end)}  = pps;
        end
        
        all_rates(neuron,angle,:) = spikes; 
        firing_rate(neuron,angle) = nanmean(spikes,2);
        error = nanstd(spikes,[],1);

    end
end

%%

angles = [30    70   110   150   190   230  ,   310   350].*(pi/180);
angles = [30    70   110   150   190   230  ,   310   350];
colors = [0 0.44 0.74; 0.8 0.2,0.2; .92 .69 .125; .2  .65 .2];


figure;
examples = 6;
rows = 3;
cols = round(6/rows);
for ex = 1 : examples
    subplot(rows,cols,ex);
    test_neuron = randi(size(data.trial(1,1).spikes,1),4,1);
    uu = 1;
    for neuron  = test_neuron'

        [line,area] = stdshade(squeeze(all_rates(neuron,:,:))',0.2,colors(uu,:),angles); hold on;
        h(uu) = line;
        f(uu) = area;
        lgd{uu} = sprintf('Neuron %i',neuron);

        uu = uu + 1;
    end
    xlabel('Angle (degree)');
    ylabel('Discharge Rates (PPS)');
    title('Tuning curve');
    legend(f(:),lgd);
end

%% 

theta_radians = deg2rad(angles);
[x, y] = pol2cart(theta_radians, 1);  
unit_vectors = [x;y];

[r_max,s_a] = max(firing_rate,[],2);
r_max = abs(r_max - mean(firing_rate, 2)); % If spike much bigger than mean-> good tuning neuron-> bigger weight
C_vec = [];
for neuron = 1 : 98
    pref_dir = theta_radians(s_a(neuron));
    [x, y] = pol2cart(pref_dir, 1);  
    C_vec = [C_vec,[x;y]];
    peak = preferred(neuron);
    fa_s(neuron,:) = r_max(neuron) .* cos((theta_radians - pref_dir));

end

%%
figure
uu = 1;
 for neuron  = [42,6,44,22]
        subplot(2,1,1);
        [line,area] = stdshade(squeeze(all_rates(neuron,:,:))',0.2,colors(uu,:),angles); hold on;
        xlabel('Angle (degree)');
        ylabel('Discharge Rates (PPS)');
        title('Tuning curve');
        subplot(2,1,2)
        [line,area] = stdshade(fa_s(neuron,:),0.2,colors(uu,:),angles); hold on;
        h(uu) = line;
        f(uu) = area;
        lgd{uu} = sprintf('Neuron %i',neuron);

        uu = uu + 1;
    end
    xlabel('Angle (degree)');
    ylabel('Tuning Response');
    title('Tuning curve');
    legend(f(:),lgd);
%%
figure;
for angle = 1 : 8
X = repmat(fa_s(:,angle),1,size(C_vec,1))';
V = X .* C_vec;
subplot(3,3,angle);
quiver(C_vec(1,:), C_vec(2,:), V(1,:),V(2,:),'k','Linewidth',1.2); hold on;
pop_vector = sum(V,2)./abs(max(sum(V,2)));
quiver(0, 0, pop_vector(1),pop_vector(2),'r','Linewidth',2);
title(sprintf('Angle %i Deg',angles(angle)));
grid on;
end
% [theta,rho] = cart2pol(V(1,:),V(2,:))
%% Visualising Population Vector 
figure;
scale_factor = 5;
for angle = 1 : 8
X = repmat(fa_s(:,angle),1,size(C_vec,1))';
V = X .* C_vec;
[x_, y_] = pol2cart(theta_radians(angle), 1); 

x_ = ones(1, size(C_vec,2)) .* scale_factor * x_;
y_ = ones(1, size(C_vec,2)) .* scale_factor * y_;

quiver(x_, y_, V(1,:),V(2,:),'k','Linewidth',1,'ShowArrowHead','on'); hold on;
pop_vector = sum(V,2)./15;
% pop_vector = sum(V,2)./abs(max(sum(V,2)));
quiver(x_(1), y_(1), pop_vector(1),pop_vector(2),'r','Linewidth',2);
% title(sprintf('Angle %i Deg',angles(angle)));
grid on;
% polarplot(theta,rho);
end

title('Neural Population Vector','FontSize',14);
%%
[sorted,I] = sort(reshape(all_rates(1,:,:),[800,1]));
y = repmat(fa_s(1,:),1,800/8)';
features = [firing_rate(1,:)',fa_s(1,:)' ];
