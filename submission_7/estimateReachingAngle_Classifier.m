function [angle] = estimateReachingAngle_Classifier(decoder,firing_rates)
% Assumption 1) each neuron only contributes to its 
% preferred orientation of movement 
        
        angle_classifier =  decoder.classifier;
        total_n_spikes = sum(firing_rates);
        
        % Training 
        F_test = firing_rates'./total_n_spikes;

        % Do not use non directional neurons for population vector decoding
        angle = angle_classifier.predictFcn(F_test);
     
end