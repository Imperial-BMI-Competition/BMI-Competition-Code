function [decoder] = angle_decoder_training(data)
    % Estiamtes Neurons Tuning curve for population vector decoding

    fsamp = 1000;
    angles = [30    70   110   150   190   230  ,   310   350];

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

%                 times_ = sum(trial(neuron,1:200));
%                 dr_ = times_ ./ (200/fsamp) ;
%                 spikes_ = [spikes_,dr];
%                 spike = find(trial(neuron,1:end)==1);
%                 pps = fsamp./diff(spike);
  
            end
%             test_rates(neuron,angle,:) = spikes_; 
            all_rates(neuron,angle,:) = spikes; 
            firing_rate(neuron,angle) = nanmean(spikes,2);
%             error = nanstd(spikes,[],1);

        end
    end
    
    % Check of neurons are directionally tuned 
    theta_radians = deg2rad(angles);
    [x, y] = pol2cart(theta_radians, 1);  
    unit_vectors = [x;y];

    [r_max,s_a] = max(firing_rate,[],2);
    directional_tuning = [];
    % r_max = (r_max - mean(firing_rate, 2)); % If spike much bigger than mean-> good tuning neuron-> bigger weight
    C_neur = [];
    for neuron = 1 : size(data(1,1).spikes,1)
        pref_dir = theta_radians(s_a(neuron));
        [x, y] = pol2cart(pref_dir, 1);  
        C_neur = [C_neur,[x;y]];
        directional_tuning(neuron) = nanstd(firing_rate(neuron,:),[],2);

        fa_s(neuron,:) = mean(firing_rate(neuron,:)) +  r_max(neuron) .* cos((theta_radians - pref_dir));

    end

    directional_threshold = 0.5;
    % directional_tuning(directional_tuning < directional_threshold) = [];

    n_dir_neurons = sum(directional_tuning < directional_threshold);
    fprintf('%i %% of neurons showed no directional tuning and were discarded.\n',round((n_dir_neurons/size(data(1,1).spikes,1)) * 100));

    discard = true;
    to_discard = directional_tuning < directional_threshold;
    
    % If wish not to discard non-directional neurons
    if ~discard
        to_discard = ones(size(r_max));
    end
    
    
    decoder = {};
    
    % Format Output
    decoder.tuning_curve = fa_s;
    decoder.preferred_angle = C_neur;
    decoder.non_directional =  to_discard;
       
end