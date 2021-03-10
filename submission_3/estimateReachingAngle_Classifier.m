function [angle] = estimateReachingAngle_Classifier(decoder,firing_rates)
% Assumption 1) each neuron only contributes to its preferred orientation of movement 
        

        angles = [30 70 110 150 190 230 310 350]; % possible angles 
     
        angle_classifier =  decoder.classifier;
        s_a = decoder.preferred_angle;
        non_directional = decoder.discard;
        total_n_spikes = sum(firing_rates);
        
         % Convert Firing rates into Features 
        F_test = []; 
      
        for angle = 1: size(angles,2)
            F_test(angle) = sum(firing_rates(s_a == angle));
        end

        % Training 
        F_test = F_test./total_n_spikes;

        
        % Do not use non directional neurons for population vector decoding
        F_test(non_directional) = [];
        
        angle = angle_classifier.predictFcn(F_test);
     
end