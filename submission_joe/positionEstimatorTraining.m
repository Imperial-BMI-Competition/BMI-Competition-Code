
function [modelParameters] = positionEstimatorTraining(training_data)

    modelParameters.Pop_Vec = SDA_decoder(training_data);
%     modelParameters.Start_Pos = avg_start_pos(training_data);

    vel = average_velocities(training_data);
    vel = average_start_pos(vel, training_data);
    vel = average_velocities_cumsum(vel, training_data);
    modelParameters.Vel = vel;
end


function [vel] = average_start_pos(vel, training_data)
    for task = 1:8
        starts = [];
        N = size(training_data,1);
        for i = 1:N
            starts = [starts training_data(i, task).handPos(:,300)];
        end
        vel(task).avg_start_pos = mean(starts, 2);
    end
end


% function [start_pos] = avg_start_pos(training_data)
%     N = size(training_data,1);
%     for i = 1:8
%         grouped_start_pos = [[]];
%         for j = 1:N
%             handPos = training_data(j,i).handPos(1:2,300);
%             grouped_start_pos = cat(3, grouped_start_pos, handPos);
%         end
%         start_pos(i).average = mean(grouped_start_pos,3); 
%     end
% end


function [velocity] = average_velocities(training_data)
    N = size(training_data,1);
    td = add_vel(training_data);
    % find average velocities
    for i = 1:8
    Grouped_coord = ([[]]);
    min = 500;
        for j = 1:N
            start = 300;
            stop = size(td(j,i).handVel,2);
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

function [vel] = average_velocities_cumsum(vel, training_data)
    for task = 1:8
        vel(task).average_cumsum = cumsum(vel(task).average,2);
    end
end
