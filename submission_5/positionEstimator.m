function [x, y, modelParameters] = positionEstimator(test_data, modelParameters) 
    
    if size(test_data.spikes,2) == 320
        modelParameters.estimated_angle = 0;
    end
    
    % Getting Firing rates estimate 
    decoder = modelParameters.Pop_Vec;
    times = sum(test_data.spikes,2); % number of spikes
    
     if size(test_data.spikes,2) <= 320
        target_id = estimateReachingAngle_Classifier(decoder,times); % estimated reaching angle
    else
        target_id = modelParameters.estimated_angle;
     end
    
%     target_id = modelParameters.true_dir;
    
    T = size(test_data.spikes,2);
    x = test_data.startHandPos(1);
    y = test_data.startHandPos(2);
    decoding_time = T - 300;
    %x = modelParameters.Start_Pos(target_id).average(1);
    %y = modelParameters.Start_Pos(target_id).average(2);
    %x = x0;
    %y = y0;

    for t = 1:decoding_time
        if t > size(modelParameters.Vel(target_id).average,2)
            % do nothing if end of the signal was reached
        else
            x = x + modelParameters.Vel(target_id).average(1,t);
            y = y + modelParameters.Vel(target_id).average(2,t);
        end
    end
    
    
    modelParameters.estimated_angle = target_id;
    
   
end



function [target_id] = get_target_id(reachingAngle)
    if (reachingAngle>=10) & (reachingAngle<50)
        target_id = 1;
    elseif (reachingAngle>=50) & (reachingAngle<90)
        target_id = 2;
    elseif (reachingAngle>=90) & (reachingAngle<130)
        target_id = 3;
    elseif (reachingAngle>=130) & (reachingAngle<170)
        target_id = 4;
    elseif (reachingAngle>=170) & (reachingAngle<210)
        target_id = 5;
    elseif (reachingAngle>=210) & (reachingAngle<270)
        target_id = 6;
    elseif (reachingAngle>=270) & (reachingAngle<330)
        target_id = 7;
    elseif (reachingAngle>=330) & (reachingAngle<360)
        target_id = 8;
    elseif (reachingAngle<10)
        target_id = 8;
    end
end