function [decoder] = SDA_decoder(data)
    % Estiamtes Neurons Tuning curve for population vector decoding

    fsamp = 1000;
    angles = [30    70   110   150   190   230     310   350];
    T_test = 320;% test on predicitng angle given the initial 320 msec only 

    %Get the dishcarge rates of all  neurons, angles and tasks
    for neuron  = 1: size(data(1,1).spikes,1)
        for angle = 1:size(data,2)
            spikes = [];
            for row = 1:size(data,1)
                T = size(data(row,angle).spikes,2);
                trial = data(row,angle).spikes;

                times = sum(trial(neuron,1:T));
                dr = times ./ (T/fsamp) ;
                spikes = [spikes,dr];

                times_ = sum(trial(neuron,1:T_test));
                n_spikes(neuron,angle,row) = times;
                n_spikes_test(neuron,angle,row) = times_;
  
            end
            firing_rate(neuron,angle) = nanmean(spikes,2);
        end
    end
    
    % Check of neurons are directionally tuned 
    theta_radians = deg2rad(angles);
    [x, y] = pol2cart(theta_radians, 1);  
    unit_vectors = [x;y];

    [r_max,s_a] = max(firing_rate,[],2);
    directional_tuning = [];
    % r_max = (r_max - mean(firing_rate, 2)); % If spike much bigger than mean-> good tuning neuron-> bigger weight
%     C_neur = [];
%     for neuron = 1 : size(data(1,1).spikes,1)
%         pref_dir = theta_radians(s_a(neuron));
%         [x, y] = pol2cart(pref_dir, 1);  
%         C_neur = [C_neur,[x;y]];
%         directional_tuning(neuron) = nanstd(firing_rate(neuron,:),[],2);
% 
%         fa_s(neuron,:) = mean(firing_rate(neuron,:)) +  r_max(neuron) .* cos((theta_radians - pref_dir));
%         
%     end

    directional_threshold = 0.5; 
    
%     n_dir_neurons = sum(directional_tuning < directional_threshold);
%     fprintf('\n%i %% of neurons showed no directional tuning and were discarded.\n',round((n_dir_neurons/size(data(1,1).spikes,1)) * 100));

    discard = true;
    to_discard = directional_tuning < directional_threshold;
    
    % If wish not to discard non-directional neurons
    if discard
        firing_rate(directional_tuning < directional_threshold,:) = [];
        s_a(directional_tuning < directional_threshold) = [];
%         fa_s(directional_tuning < directional_threshold,:) = [];
        n_spikes(directional_tuning < directional_threshold,:,:) = [];
        n_spikes_test(directional_tuning < directional_threshold,:,:) = [];
    end
     % Build Fetures Vector 
    F = []; F_ = [];
    y_true = []; y_true_ = [];

    F_test = [];
    y_true_test = [];

    for i = 1: size(n_spikes,2)
        for j = 1:  size(n_spikes,3)
            total_n_spikes = sum(n_spikes(:,i,j));
            total_n_spikes_test = sum(n_spikes_test(:,i,j));

            f = []; f_test =  [];
            for angle = 1: size(n_spikes,2)
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
            
        end
    end
    
    F_all = [[F, y_true];[F_test,y_true_test]];
    [angle_classifier, validationAccuracy] = trainClassifierSubSpace(F_all);
%     fprintf('\nReaching angle decoder (Subspace Discriminant) Validation Accuracy %2.1f %%',100*validationAccuracy);
    decoder = {};
    
    % Format Output
    decoder.model = 'Subspace Discriminant';
    decoder.classifier = angle_classifier;
    decoder.accuracy =  validationAccuracy;
    decoder.preferred_angle = s_a;
    decoder.discard = directional_tuning < directional_threshold;
   
       
end