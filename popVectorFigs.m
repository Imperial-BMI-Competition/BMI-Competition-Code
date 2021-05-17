%Pre competition work
% Manfredi Castelli 
% 01/03/2021
clc;
clear all;
close all;
warning('off');
% data = load('monkeydata_training.mat');
data = load('../monkeydata0.mat');

%% Peri-stimulus time histograms
%Histogram of the times at which a neuron fires. One histogram per neural
%unit.
tic;
fsamp = 1000;
angle = 1;
neuron = 5;
inst_rates = {};
angles = [30    70   110   150   190   230  ,   310   350];
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
            spikes_ = [spikes_,dr_];
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
    C_neur = [C_neur,[x;y]]; % neuron preferred direction in cartesian coordinates 
%     peak = preferred(neuron);
    directional_tuning(neuron) = nanstd(firing_rate(neuron,:),[],2);

    fa_s(neuron,:) = mean(firing_rate(neuron,:)) +  r_max(neuron) .* cos((theta_radians - pref_dir));

end

directional_threshold = 1;
% directional_tuning(directional_tuning < directional_threshold) = [];

n_dir_neurons = sum(directional_tuning < directional_threshold);
fprintf('%i %% of neurons showed no directional tuning and were discarded.\n',round((n_dir_neurons/98) * 100));

firing_rates_valid = firing_rate(directional_tuning > directional_threshold,:);

[r_max,s_a_valid] = max(firing_rates_valid,[],2); % get neuron preferred direction 
toc
figure; 
h = histogram(s_a,'Normalization','pdf');
distribution = h.Values;
xlabel('Preferred Angle (ith angle)')
ylabel('Count','Fontsize',14)
xlabel('Preferred Angle (ith angle)','Fontsize',14);
hold on;
histogram(s_a_valid,'Normalization','pdf');
legend('All Neurons','Directional Neurons Only');
grid on;
title('Distribution of Neuron Preferred Directions over all trials','Fontsize',14)
discard = true;
 
    %% Visualising Population Vector 
figure;
scale_factor = 5; % scale vector just for visualisation 
for angle = 1 : 8
X = repmat(fa_s(:,angle),1,size(C_neur,1))';
V = X .* C_neur;
[x_, y_] = pol2cart(theta_radians(angle), 1); % convert angle to a cartesian vector

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
rmse = [];
y_pred = [];
y_true = [];
for test_angle = 1: 8
    for test_trial = 1:100
        
        features = (test_rates(:,test_angle,test_trial) - mean_firing)./r_max; % weight of each neuron
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
    quiver(0, 0, 0.3.*pop_vector(1),0.3.*pop_vector(2),'r','Linewidth',2);
    hold on;
    quiver(0, 0, 20 *unit_vectors(1,test_angle),20*unit_vectors(2,test_angle),'g','Linewidth',2);
    
    title(sprintf('Population Vector: Tested Trial % i - Angle %i Deg',test_trial,angles(test_angle)));
    grid on;axis equal
    legend({'Individual Response','Estimated Direction','True Direction'})
    end
    alpha = vectors_angle(pop_vector,unit_vectors(:,test_angle));
    rmse = [rmse; [(pop_vector(2)/pop_vector(1)) ,theta_radians(test_angle)]];
    labelRad = cart2pol(pop_vector(1),pop_vector(2));
    if labelRad<0  labelRad = 2*pi +labelRad;end
    [~,label] = min(abs( labelRad - theta_radians));
    y_pred = [y_pred label];
    y_true = [y_true test_angle];
    error_angle(test_angle,test_trial) = 1 - (pop_vector(2)/pop_vector(1))/theta_radians(test_angle) ;
    error_angle(test_angle,test_trial) = alpha;
%     error_angle(test_angle,test_trial) = (pop_vector(2)/pop_vector(1)) ;
%     true_angle(test_angle,test_trial) = theta_radians(test_angle);
%     error_angle(test_angle,test_trial) = 180/pi * abs((pop_vector(2)/pop_vector(1)) - theta_radians(test_angle) );
    end
end
 mean(rms(rmse,2))
% [angles_mesh, trial_mesh ] = meshgrid(angles,1:100);
% 
% figure; 
% surf(angles_mesh,trial_mesh,abs(error_angle)');
% xlabel('True Angle (deg)','Fontsize',14);
% ylabel('Trial','Fontsize',14)
% zlabel('Angle between true and estimated vector (deg)','Fontsize',14);

figure; 
boxplot(error_angle');
xticklabels(angles);
xlabel('True Angle (deg)','Fontsize',14);
ylabel('Angle (deg)','Fontsize',14);
title('Angle between True and Estimated direction','Fontsize',14);
grid on;

figure; confusionchart(y_true,y_pred);