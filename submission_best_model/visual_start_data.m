load ../monkeydata0.mat

for ta = 1:8
    figure
    for tr = 1:100
        data = trial(tr, ta);
%         handtraj = data.handPos(:, :);
%         plot3(handtraj(1,:), handtraj(2,:), handtraj(3,:))
%         hold on

        vel = add_vel(data);
        vel = vecnorm(vel.handVel, 1);
        
        plot(vel);
        hold on;
    end
end
    