function td = add_vel(trial_data)
% Function that adds a velocity field to each trial in the monkey_data 
% new field name = handVel
% Assumes that the input data includes field named handPos
    td = trial_data;
    [M, N] = size(td);
    % Looping through all trials
    for m = 1:M
        for n = 1:N
            % Total number of time steps for current trial
            T = size(td(m,n).handPos, 2); 
            % Initialisation of velocity matrix
            vel = zeros(3,T-1);
            % Find velocity for current trial (difference in position)
            for t = 2:T
                vel(:,t-1) = td(m,n).handPos(:,t) - td(m,n).handPos(:,t-1);
            end
            % Add the handVel field to the struct with velocity data
            td(m,n).handVel = deal(vel);
        end
    end
end
