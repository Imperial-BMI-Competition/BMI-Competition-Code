load ../monkeydata0.mat

for ta = 1:8
    figure
    for tr = 1:100
        data = trial(tr, ta);
        handtraj = data.handPos(:, 1:300);
        plot3(handtraj(1,:), handtraj(2,:), handtraj(3,:))
        hold on
    end
end
    