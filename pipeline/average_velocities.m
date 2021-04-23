function [velocity] = average_velocities(training_data)
    N = size(training_data,1);
    td = add_vel(training_data);
    % find average velocities
    for i = 1:8
    Grouped_coord = ([[]]);
    min = 500;
        for j = 1:N
            start = 300;
            stop = size(td(j,i).handVel,2)-100;
           if (stop-start)<min
              min = stop - start; 
           end
        end
        for j = 1:N
           start = 300;
           stop = start + min;

           coord = td(j,i).handVel(:, start:stop);
           Grouped_coord = cat(3, Grouped_coord, coord);
        end
        velocity(i).average = mean(Grouped_coord,3);
    end
end
