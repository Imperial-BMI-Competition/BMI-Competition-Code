%Pre competition work
% Manfredi Castelli 
% 01/03/2021
clc;
clear all;
close all;

data = load('monkeydata_training.mat');
trials = data.trial;
trial1 = trials(1,1).handPos;
trial2 = trials(2,5).handPos;
%%
figure()
plot(trial1(1,:),trial1(2,:))
hold on 
plot(trial2(1,:),trial2(2,:))

legend("Trial 1", "Trial 2")

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
scatter(y,x,5,'filled'); % int indicates size of dots
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
    scatter(x{1,trials}, y{1,trials},'linewidth',1); %each trial is a different colour
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
        spikes_ =  [];
        for row = 1:100
            T = size(data.trial(row,angle).spikes,2);
            trial = data.trial(row,angle).spikes;
            times = sum(trial(neuron,1:T));
            dr = times ./ (T/fsamp) ;
            spikes = [spikes,dr];
            
            times_ = sum(trial(neuron,1:300));
            dr_ = times_ ./ (300/fsamp) ;
            spikes_ = [spikes_,dr];
            spike = find(trial(neuron,1:end)==1);
            pps = fsamp./diff(spike);
%             inst_rates{neuron,angle,row} = trial(neuron,1:200) ; 
%             inst_rates{neuron,angle,row,spike(2:end)}  = pps;
        end
        test_rates(neuron,angle,:) = spikes_; 
        all_rates(neuron,angle,:) = spikes; 
        firing_rate(neuron,angle) = nanmean(spikes,2);
        error = nanstd(spikes,[],1);

    end
end
%%
figure;
angles = [30    70   110   150   190   230  ,   310   350];
neur = 44;
lgd = {};
for alpha = 1:8

    histogram(all_rates(neur,alpha,:)); hold on;
    xlabel('Firing rate (pps)','Fontsize',14);
    lgd{alpha} = sprintf('Angle %i',angles(alpha));
end
legend(lgd);
%% Population Vector Analysis 
% Get Firing rate of individual neruons 
% avg across trials 
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

%% Get TUning curve of each neuron based on its discharge rate as function of reaching angle

theta_radians = deg2rad(angles);
[x, y] = pol2cart(theta_radians, 1);  
unit_vectors = [x;y];

[r_max,s_a] = max(firing_rate,[],2);
directional_tuning = [];
% r_max = (r_max - mean(firing_rate, 2)); % If spike much bigger than mean-> good tuning neuron-> bigger weight
C_neur = [];
for neuron = 1 : 98
    pref_dir = theta_radians(s_a(neuron));
    [x, y] = pol2cart(pref_dir, 1);  
    C_neur = [C_neur,[x;y]];
%     peak = preferred(neuron);
    directional_tuning(neuron) = nanstd(firing_rate(neuron,:),[],2);

    fa_s(neuron,:) = mean(firing_rate(neuron,:)) +  r_max(neuron) .* cos((theta_radians - pref_dir));

end

directional_threshold = 1;
% directional_tuning(directional_tuning < directional_threshold) = [];

n_dir_neurons = sum(directional_tuning < directional_threshold);
fprintf('%i %% of neurons showed no directional tuning and were discarded.\n',round((n_dir_neurons/98) * 100));

firing_rates_valid = firing_rate(directional_tuning > directional_threshold,:);

[r_max,s_a_valid] = max(firing_rates_valid,[],2);

figure; 
h = histogram(s_a,'Normalization' ,'probability');
distribution = h.Values;
xlabel('Preferred Angle (ith angle)')
ylabel('Count','Fontsize',14)
xlabel('Preferred Angle (ith angle)','Fontsize',14);
hold on;
histogram(s_a_valid);
legend('All Neurons','Directional Neurons Only');

discard = true;


%% Compare raw tuning curve and smoothed one with cos 
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
%% Plot vector field 
figure;
for angle = 1 : 8
X = repmat(fa_s(:,angle),1,size(C_neur,1))';
V = X .* C_neur;
subplot(3,3,angle);
quiver(C_neur(1,:), C_neur(2,:), V(1,:),V(2,:),'k','Linewidth',1.2); hold on;
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
X = repmat(fa_s(:,angle),1,size(C_neur,1))';
V = X .* C_neur;
[x_, y_] = pol2cart(theta_radians(angle), 1); 

x_ = ones(1, size(C_neur,2)) .* scale_factor * x_;
y_ = ones(1, size(C_neur,2)) .* scale_factor * y_;

quiver(x_, y_, V(1,:),V(2,:),'k','Linewidth',1,'ShowArrowHead','on'); hold on;
pop_vector = sum(V,2)./13;
% pop_vector = sum(V,2)./abs(max(sum(V,2)));
quiver(x_(1), y_(1), pop_vector(1),pop_vector(2),'r','Linewidth',2);
% title(sprintf('Angle %i Deg',angles(angle)));
grid on;
% polarplot(theta,rho);
end

title('Neural Population Vector','FontSize',14);

%% Regression of tuning curves 

models = [];
for neuron = 1 : 98
    y = firing_rate(neuron,:);
    x = angles;
    p = polyfit(x,y,3);
    
    models(neuron,:) = p;
    x1 = 10:5:350;
    y1 = polyval(p,x1);
    
    models_tuning(neuron,:) = y1; 
end

%% Test Model 
% Select Trial and angle to test
test_trial = 100;
test_angle = 1;


error_angle = zeros(8,100);

if discard
    test_rates(directional_tuning < directional_threshold,:,:) = [];
    firing_rate(directional_tuning < directional_threshold,:) = [];
    s_a(directional_tuning < directional_threshold) = [];
    C_neur(:,directional_tuning < directional_threshold) = [];
    fa_s(directional_tuning < directional_threshold,:) = [];
end

max_firing = max(firing_rate,[],2);
min_firing = min(firing_rate,[],2);
mean_firing = mean(fa_s,2);
show_plot = false;
%%
tollerance = 1;
for test_angle = 1: 8
    for test_trial = 1:100
        
        features = test_rates(:,test_angle,test_trial) - mean_firing; % weight of each neuron
        if tollerance
            within = abs(test_rates(:,test_angle,test_trial) - mean_firing) < tollerance;
            features(within) = abs(features(within));
            
        end
%         low_count_neurons  = find(distribution < 0.1);
        [low_count_neurons, ~] = ismember(s_a, find(distribution > 0.1));
        features(low_count_neurons) = features(low_count_neurons) .* 0.7;
        Weights = repmat(features,1,size(C_neur,1))';
        N = Weights .* C_neur; % weighted individual directions 
        
        pop_vector = sum(N,2); % population vector
    
    if show_plot 
    figure;
    origin = zeros(1,size(N,2));
    quiver(origin, origin, N(1,:),N(2,:),'k','Linewidth',1); hold on;
    quiver(0, 0, pop_vector(1),pop_vector(2),'r','Linewidth',2);
    hold on;
    quiver(0, 0, 20 *unit_vectors(1,test_angle),20*unit_vectors(2,test_angle),'g','Linewidth',2);
    grid on;
    title(sprintf('Tested Trial % i - Angle %i Deg',test_trial,angles(test_angle)));
    end
    alpha = vectors_angle(pop_vector,unit_vectors(:,test_angle));

    error_angle(test_angle,test_trial) = 1 - (pop_vector(2)/pop_vector(1))/theta_radians(test_angle) ;
    error_angle(test_angle,test_trial) = alpha;
%     error_angle(test_angle,test_trial) = (pop_vector(2)/pop_vector(1)) ;
%     true_angle(test_angle,test_trial) = theta_radians(test_angle);
%     error_angle(test_angle,test_trial) = 180/pi * abs((pop_vector(2)/pop_vector(1)) - theta_radians(test_angle) );
    end
end

% [angles_mesh, trial_mesh ] = meshgrid(angles,1:100);
% 
% figure; 
% surf(angles_mesh,trial_mesh,abs(error_angle)');
% xlabel('True Angle (deg)','Fontsize',14);
% ylabel('Trial','Fontsize',14)
% zlabel('Angle between true and estimated vector (deg)','Fontsize',14);

figure; 
stdshade(error_angle',0.2,colors(2,:),angles);
xlabel('True Angle (deg)','Fontsize',14);
ylabel('Angle (deg)','Fontsize',14);
title('Angle between True and Estimated direction','Fontsize',14);

% for i = 1:8
%   rmse(i) = sqrt(immse(true_angle(i,:),error_angle(i,:)));
% end
% figure; 
% stdshade(rmse,0.2,colors(2,:),angles);
% xlabel('True Angle (deg)','Fontsize',14);
% ylabel('RMSE (deg)','Fontsize',14);
% title(sprintf('Avg. Testing RMSE  between True and Estimated direction %1.2f',mean(rmse)),'Fontsize',14);
%%
figure; 
plot(models_angle,models_tuning(neuron,:),'Linewidth',2), hold on;
yline(features(neuron),'r','Linewidth',2)
%%
neuron = 84;
figure;
plot(angles,fa_s(neuron,:),'Linewidth',2), hold on;
yline((test_rates(neuron,test_angle,test_trial) ),'r','Linewidth',2);
yline(mean_firing(neuron),'k--','Linewidth',2)

%% Construct NN data set 
clc;
data_set = [];
label = [];
norm_rates = normalize(all_rates,1,'range');
max_factor = max(all_rates,[],'all');
for n = 1:98
    for angle = 1:8
        
        for trial = 1:100
            data_set = [data_set; norm_rates(:,trial)',angle];
            label = [label; angle];
        end
    end
end