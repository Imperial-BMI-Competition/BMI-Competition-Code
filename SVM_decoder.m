% SVM Classificiation
% Manfredi Castelli 
% 06/03/2021
clc;
clear all;
close all;

data = load('monkeydata_training.mat');
trials = data.trial;
trial1 = trials(1,1).handPos;
trial2 = trials(2,5).handPos;



%% Tuning FUncitons for labelling neurons 
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
%             T = size(data.trial(row,angle).spikes,2);
            T = size(data.trial(row,angle).spikes,2);
            trial = data.trial(row,angle).spikes;
            times = sum(trial(neuron,1:T));
            dr = times ./ (T/fsamp) ;
            spikes = [spikes,dr];
            
            times_ = sum(trial(neuron,1:320));
            dr_ = times_ ./ (320/fsamp) ;
            spikes_ = [spikes_,dr];
            spike = find(trial(neuron,1:end)==1);
            pps = fsamp./diff(spike);
             
            n_spikes(neuron,angle,row) = times;
            n_spikes_test(neuron,angle,row) = times_;
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
% lgd = {};
% for alpha = 1:8
% 
%     histogram(all_rates(neur,alpha,:)); hold on;
%     xlabel('Firing rate (pps)','Fontsize',14);
%     lgd{alpha} = sprintf('Angle %i',angles(alpha));
% end
% legend(lgd);
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

directional_threshold = 0.5;
% directional_tuning(directional_tuning < directional_threshold) = [];

n_dir_neurons = sum(directional_tuning < directional_threshold);
fprintf('%i %% of neurons showed no directional tuning and were discarded.\n',round((n_dir_neurons/98) * 100));

firing_rates_valid = firing_rate(directional_tuning > directional_threshold,:);

[r_max,s_a_valid] = max(firing_rates_valid,[],2);

% figure; histogram(s_a);
% xlabel('Preferred Angle (ith angle)')
% ylabel('Count','Fontsize',14)
% xlabel('Preferred Angle (ith angle)','Fontsize',14);
% hold on;
% histogram(s_a_valid);
% legend('All Neurons','Directional Neurons Only');

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




%% Test Model 
% Select Trial and angle to test
test_trial = 100;
test_angle = 1;


error_angle = zeros(8,100);

if discard
    test_rates(directional_tuning < directional_threshold,:,:) = [];
    firing_rate(directional_tuning < directional_threshold,:) = [];
    s_a(directional_tuning < directional_threshold) = [];
%     C_neur(:,directional_tuning < directional_threshold) = [];
    fa_s(directional_tuning < directional_threshold,:) = [];
    n_spikes(directional_tuning < directional_threshold,:,:) = [];
    n_spikes_test(directional_tuning < directional_threshold,:,:) = [];
end

max_firing = max(firing_rate,[],2);
min_firing = min(firing_rate,[],2);
mean_firing = mean(fa_s,2);
show_plot = false;

%% Build Features Vector
F = []; F_ = [];
y_true = []; y_true_ = [];

F_test = [];
y_true_test = [];

for i = 1: 8
    for j = 1: 100
        total_n_spikes = sum(n_spikes(:,i,j));
        total_n_spikes_test = sum(n_spikes_test(:,i,j));
        
        f = []; f_test =  [];
        for angle = 1:8
            f(angle) = sum(n_spikes(s_a == angle,i,j));
            f_test(angle) =  sum(n_spikes_test(s_a == angle,i,j));
        end
        
        % Training 
        f = f./total_n_spikes;
        F = [F;f]; 
        y_true = [y_true; i];
        
        % Testing - 200 msec only
        f_test = f_test./total_n_spikes_test;
        F_test = [F_test;f_test]; 
        y_true_test = [y_true_test; i];
        
%         % Final Testing - 200 msec only
%         f_test = f_test./total_n_spikes_test;
%         F_test = [F_test;f_test]; 
%         y_true_test = [y_true_test; i];
    end
end
%%
cv = cvpartition(size(F_test,1),'HoldOut',0.3);
idx = cv.test;

testing_elem = randperm(800,500);
F_all = [[F, y_true];[F_test(idx,:),y_true_test(idx,:)]];
[angle_classifier, validationAccuracy] = trainClassifier(F_all);




% Testing on 200 msec data

ypred = angle_classifier.predictFcn(F_test(~idx,:));
figure;
cm = confusionchart(y_true_test(~idx,:),ypred)
title(sprintf('QDA PCA - Accuracy %2.1f %%- MSE %2.2f',100 * sum(y_true_test(~idx,:) == ypred)/size(ypred,1),immse(y_true_test(~idx,:),ypred)));
%%
tollerance = 1;
for test_angle = 1: 8
    for test_trial = 1:100
        
        features = test_rates(:,test_angle,test_trial) - mean_firing; % weight of each neuron
        if tollerance
            within = abs(test_rates(:,test_angle,test_trial) - mean_firing) < tollerance;
            features(within) = abs(features(within));
            
        end
        Weights = repmat(features,1,size(C_neur,1))';
        N = Weights .* C_neur; % weighted individual directions 
        
        pop_vector = sum(N,2); % population vector
    
    if show_plot 
    figure;
    origin = zeros(1,size(V,2));
    quiver(origin, origin, V(1,:),V(2,:),'k','Linewidth',1); hold on;
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

