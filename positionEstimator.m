function [x, y] = positionEstimator(test_data, modelParameters)

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
   
    %LSTM_param = modelParameters.LSTM;
    decoder = modelParameters.Pop_Vec;
    
    fsamp = 1000; % sampling Frequency is 1 sample/msec
    
    
    trials = test_data;
    trials.handPos = [trials.startHandPos, trials.decodedHandPos];
    
    size_t = length(trials); %number of trials
    t = size(trials.spikes);  %split spikes into 20 sec windows and take average over them
    spikes = [];
    for i = 300:20:t(2)-20
        s = mean(trials.spikes(:,i:i+20), 2); %take mean along the rows
        spikes = [spikes, s];
    end
    
    % Getting Firing rates estimate 
    T = size(test_data.spikes(:,1:300),2);
    trial = test_data.spikes(:,1:300);
    
    times = sum(trial,2); % number of spikes
    all_rates = times ./ (T/fsamp); % firing rates in pulse per second (pps)
    
    
    [population_vector, reachingAngle] = estimateReachingAngle(decoder,all_rates); % estimated reaching angle
    %dataTest = [trials.handPos]; %use neural spikes as feature as well as position
    
    % from the reachingAngle get target_id (from 1 to 8 - one for each target)
    target_id = get_target_id(reachingAngle);
    
    
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
    %Standardise all the features by save values as training set
    %test_mu = LSTM_params.mu;
    %test_sig = LSTM_params.sig;

    %dataTest_standardised = (dataTest - test_mu)./ (test_sig +0.0000001); %so it never divides by 0

    %XTest = dataTest_standardised;
    %YPred = XTest; %initialise as the same

    %net = LSTM_params.net;

    %s = size(XTest);
    %numTimeStepsTest = s(2);
    
    %for i = 2:numTimeStepsTest
    %   [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i-1),'ExecutionEnvironment','cpu');
    %end
    
%     newModelParameters.net = net;
%     newModelParameters.mu = test_mu;
%     newModelParameters.sig = test_sig;

    % Unnormalise predictions
   % YPred_units = test_sig.*YPred + test_mu;

    % for t = 1:length(YPred) % number of test trials
    %     y = YPred(t);
    %     y = test_sig.*y + test_mu;
    %     YPred_units= [YPred_units; y];    
    % end

   % predict = YPred_units(:,end);
   % x = predict(1);
   % y = predict(2);   
end

function [pop_vector,angle] = estimateReachingAngle(decoder,firing_rates)
% Assumption 1) each neuron only contributes to its preferred orientation of movement 
        

        angles = [30 70 110 150 190 230 310 350]; % possible angles 
        theta_radians = deg2rad(angles); % convert angles to radians 
        
        tollerance = 1; % parameter regulating the tollerance/sensitivity of decoding 
                        % of poulation vector 

        fa_s = decoder.tuning_curve;  % tuning curves
        C_neur = decoder.preferred_angle; % preferred direction of each neuron
        [~,s_a] = max(fa_s,[],2); % preferred discretised angle of each neuron
        
        mean_firing = mean(fa_s,2); % mean firing rates of each neuron 
                                    % if test > mean_firing then
                                    % directional angle is in the preferred
                                    % direction; if test < mean_firing then
                                    % then directional angle is oppositive
                                    % to preferred direction 
        
        % Do not use non directional neurons for population vector decoding
        firing_rates(decoder.non_directional) = [];
        mean_firing(decoder.non_directional) = [];
        s_a(decoder.non_directional) = [];
        C_neur(:,decoder.non_directional) = [];
        fa_s(decoder.non_directional,:) = [];
        
        % get discharge rates features for poulation decoding
        features = firing_rates - mean_firing; % weight of each neuron
        
        if tollerance  % if seems to encode opppositive direction to small extent ( 0 > features > -1) adjust to positive (preferred direction
            within = abs(features) < tollerance;
            features(within) = abs(features(within));
            
        end
        Weights = repmat(features,1,size(C_neur,1))';
        N = Weights .* C_neur; % weighted individual directions 
        
        pop_vector = sum(N,2); % population vector
        %angle = (pop_vector(2)/pop_vector(1)) * 180/pi;
        angle = mod(cart2pol(pop_vector(1), pop_vector(2)),2*pi) * 180/pi;
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