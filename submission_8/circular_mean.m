function [circ_mean_idx] = circular_mean(values)
    original_angles = [30    70   110   150   190   230     310   350];
    angles = values_to_angles(values);
    circ_mean = mean_angle(angles);
    
    diff_1 = abs(original_angles - circ_mean);
    diff_2 = abs(original_angles - circ_mean - 360);
    
    [~,I] = min([diff_1 diff_2]);
    if I > 8
        I = I - 8;
    end
    circ_mean_idx = I;
end


function [values] = values_to_angles(values)
    angles = [30    70   110   150   190   230     310   350];
    for i = 1:size(values,2)
        values(i) = angles(values(i));
    end
end


function u = mean_angle(phi)
    % obtained from https://rosettacode.org/wiki/Averages/Mean_angle#:~:text=When%20calculating%20the%20average%20or,measure%20of%20the%20same%20angle.&text=Convert%20the%20complex%20mean%20to,is%20the%20required%20angular%20mean.
	u = angle(mean(exp(i*pi*phi/180)))*180/pi;
end