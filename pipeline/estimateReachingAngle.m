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
