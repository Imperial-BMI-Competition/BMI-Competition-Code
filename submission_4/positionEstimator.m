function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  
  
  
  % ... compute position at the given timestep.
  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
  
    
    if size(test_data.spikes,2) == 320
        modelParameters.estimated_angle  = 0;
    end
    
    % Getting Firing rates estimate 
    decoder = modelParameters.Pop_Vec;
    times = sum(test_data.spikes,2); % number of spikes
    
     if size(test_data.spikes,2) <= 320
        target_id = estimateReachingAngle_Classifier(decoder,times); % estimated reaching angle
    else
        target_id = modelParameters.estimated_angle;
    end
    
    
    T = size(test_data.spikes,2);
    x0 = test_data.startHandPos(1);
    y0 = test_data.startHandPos(2);
    decoding_time = T - 300;
    x = x0;
    y = y0;

               
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