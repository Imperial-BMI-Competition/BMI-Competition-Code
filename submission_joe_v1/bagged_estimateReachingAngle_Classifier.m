function [angles] = bagged_estimateReachingAngle_Classifier(decoders,F_test)
    % Assumption 1) each neuron only contributes to its 
    % preferred orientation of movement 

    angles = [];
    s = size(decoders,2);
    for i = 1:s
        angle_classifier =  decoders(i).classifier;

        % Do not use non directional neurons for population vector decoding
        angle = angle_classifier.predictFcn(F_test);
        angles = [angles angle];
    end
end