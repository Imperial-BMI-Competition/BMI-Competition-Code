function [angle] = bagged_estimateReachingAngle_Classifier(decoders,firing_rates)
% Assumption 1) each neuron only contributes to its 
% preferred orientation of movement 
        
    total_n_spikes = sum(firing_rates);
    F_test = firing_rates'./total_n_spikes;

    angles = [];
    s = size(decoders,2);
    for i = 1:s
        angle_classifier =  decoders(i).classifier;

        % Do not use non directional neurons for population vector decoding
        angle = angle_classifier.predictFcn(F_test);
        angles = [angles angle];
    end
    angle = mode(angles);
end