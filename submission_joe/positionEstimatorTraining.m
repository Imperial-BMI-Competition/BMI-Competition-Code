
function [modelParameters] = positionEstimatorTraining(training_data)
    Pop_Vecs = [];
    tr_s = size(training_data,1);
    for i = 1:3
        idxes = randperm(tr_s, int8(tr_s * 0.95));
        idxes = sort(idxes);
        tr_x = training_data(idxes,:);
        
        Pop_Vecs = [Pop_Vecs SDA_decoder(tr_x, 1, 320)];
    end
    modelParameters.Pop_Vec = Pop_Vecs;
    
    Pop_Vecs = [];
    tr_s = size(training_data,1);
    for i = 1:0
        idxes = randperm(tr_s, int8(tr_s * 0.95));
        idxes = sort(idxes);
        tr_x = training_data(idxes,:);
        Pop_Vecs = [Pop_Vecs SDA_decoder(tr_x, 320, 440)];
    end
    modelParameters.Pop_Vec_Mid = Pop_Vecs;
    
    accs = [];
    for i = 1:size(modelParameters.Pop_Vec,2)
        accs = [accs modelParameters.Pop_Vec(i).accuracy];
    end
    fprintf('\nReaching angle decoder (Quadratic Discriminant) Validation Accuracy %2.1f %%\n',100*mean(accs));
    disp(accs)
    accs = [];
    for i = 1:size(modelParameters.Pop_Vec_Mid,2)
        accs = [accs modelParameters.Pop_Vec_Mid(i).accuracy];
    end
    fprintf('\nReaching angle decoder (Quadratic Discriminant) Validation Accuracy %2.1f %%\n',100*mean(accs));

    vel = average_velocities(training_data);
    vel = average_start_pos(vel, training_data);
    vel = average_velocities_cumsum(vel, training_data);
    modelParameters.Vel = vel;
    modelParameters.mid_estimated_angle = 0;
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


function [velocity] = average_velocities(training_data)
    N = size(training_data,1);
    td = add_vel(training_data);
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
