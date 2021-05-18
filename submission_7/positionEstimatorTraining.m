
function [modelParameters] = positionEstimatorTraining(training_data, modelParameters)
    modelParameters.mid_estimated_angle = 0;

    
    
    [~, avg_rate] = get_firing_ratio(training_data, 1, modelParameters.init_idx, ...
                                                modelParameters.hold_out, ...
                                                modelParameters.time_norm, modelParameters.mean_adjust, modelParameters.max_norm);
    modelParameters.avg_rate = avg_rate;
                                  
    Pop_Vecs = [];
    tr_s = size(training_data,1);
    for i = 1:modelParameters.init_bag_size
        idxes = randperm(tr_s, int8(tr_s * modelParameters.bag_data_split));
        idxes = sort(idxes);
        tr_x = training_data(idxes,:);
        [Firing_rates, ~] = get_firing_ratio(tr_x, 1, modelParameters.init_idx, ...
                                        modelParameters.hold_out, ...
                                        modelParameters.time_norm, modelParameters.mean_adjust, modelParameters.max_norm);
        Pop_Vecs = [Pop_Vecs QD_decoder(Firing_rates, modelParameters.validateQD, modelParameters.discrimType, modelParameters.pc_components)];
    end
    modelParameters.Pop_Vec = Pop_Vecs;
    
    Pop_Vecs = [];
    tr_s = size(training_data,1);
    for i = 1:modelParameters.mid_bag_size
        idxes = randperm(tr_s, int8(tr_s * modelParameters.bag_data_split));
        idxes = sort(idxes);
        tr_x = training_data(idxes,:);
        [Firing_rates, ~] = get_firing_ratio(tr_x, 1, modelParameters.init_idx, ...
                                        modelParameters.hold_out, ...
                                        modelParameters.time_norm, modelParameters.mean_adjust, modelParameters.max_norm);
        Pop_Vecs = [Pop_Vecs QD_decoder(Firing_rates, modelParameters.validateQD, modelParameters.discrimType, modelParameters.pc_components)];
    end
    modelParameters.Pop_Vec_Mid = Pop_Vecs;
    
    if modelParameters.validateQD
        accs = [];
        for i = 1:size(modelParameters.Pop_Vec, 2)
            accs = [accs modelParameters.Pop_Vec(i).accuracy];
        end
        fprintf('\nReaching angle decoder (Quadratic Discriminant) Validation Accuracy %2.1f %%\n',100*mean(accs));
        disp(accs)
        accs = [];
        for i = 1:size(modelParameters.Pop_Vec_Mid, 2)
            accs = [accs modelParameters.Pop_Vec_Mid(i).accuracy];
        end
        fprintf('\nReaching angle decoder (Quadratic Discriminant) Validation Accuracy %2.1f %%\n',100*mean(accs));
    end

    vel = average_velocities(training_data);
    vel = average_start_pos(vel, training_data);
    vel = average_velocities_cumsum(vel);
    modelParameters.Vel = vel;
    
    if modelParameters.tree_freq ~= 0
        modelParameters.tree = tree_ensemble(training_data, ...
            modelParameters.tree_filter_length, modelParameters.tree_sum_length, modelParameters.tree_binsize);
    end
end


function [vel] = average_start_pos(vel, training_data)
    for task = 1:8
        starts = [];
        N = size(training_data,1);
        for i = 1:N
            starts = [starts training_data(i, task).handPos(:, 300)];
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
            stop = size(td(j,i).handVel, 2);
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

function [vel] = average_velocities_cumsum(vel)
    for task = 1:8
        vel(task).average_cumsum = cumsum(vel(task).average,2);
    end
end
