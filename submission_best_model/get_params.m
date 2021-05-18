function [modelParameters] = get_params(modelParameters)

    if modelParameters.model_name == "QDA w/ Avg. Traj. w/ Convergence"
        modelParameters.init_idx = 320;
        modelParameters.init_bag_size = 1;
        modelParameters.mid_idx = 360;
        modelParameters.mid_bag_size = 1;
        modelParameters.bag_data_split = 0.99;
        modelParameters.w = 0.9;
        modelParameters.hold_out = 0.8;
        modelParameters.validateQD = false;
        modelParameters.tree_filter_length = 0; % 0, 1, 2
        modelParameters.tree_sum_length = 0; % 0, 1, 2
        modelParameters.tree_binsize = 0; % 40, 80, 120
        modelParameters.tree_freq = 0; % 160, 320
        modelParameters.start_floor_binsize = floor(300 / modelParameters.tree_binsize);
        modelParameters.discrimType = 'quadratic'; %['linear' 'quadratic' 'pseudoquadratic']
        modelParameters.pc_components = 10;
        modelParameters.time_norm = false;
        modelParameters.mean_adjust = true;
        modelParameters.max_norm = true;
        modelParameters.use_average_traj = false;
        modelParameters.use_true = false;
        
    elseif modelParameters.model_name == "QDA w/ Avg. Traj."
        modelParameters.init_idx = 320;
        modelParameters.init_bag_size = 1;%4;
        modelParameters.mid_idx = 360;
        modelParameters.mid_bag_size = 1;%8;
        modelParameters.bag_data_split = 0.7;
        modelParameters.w = 0.0;
        modelParameters.hold_out = 0.5;
        modelParameters.validateQD = false;
        modelParameters.tree_filter_length = 0; % 0, 1, 2
        modelParameters.tree_sum_length = 0; % 0, 1, 2
        modelParameters.tree_binsize = 0; % 40, 80, 120
        modelParameters.tree_freq = 0;
        modelParameters.start_floor_binsize = floor(300 / modelParameters.tree_binsize);
        modelParameters.discrimType = 'quadratic'; %['linear' 'quadratic' 'pseudoquadratic']
        modelParameters.pc_components = 40;
        modelParameters.time_norm = false;
        modelParameters.mean_adjust = true;
        modelParameters.max_norm = true;  
        modelParameters.x_start_path = false;
        modelParameters.use_true = false;
        
     elseif modelParameters.model_name == "QDA w/ Avg. Traj. w/ Avg. Start"
        modelParameters.init_idx = 320;
        modelParameters.init_bag_size = 1;%4;
        modelParameters.mid_idx = 360;
        modelParameters.mid_bag_size = 1;%8;
        modelParameters.bag_data_split = 0.7;
        modelParameters.w = 0.0;
        modelParameters.hold_out = 0.5;
        modelParameters.validateQD = false;
        modelParameters.tree_filter_length = 0; % 0, 1, 2
        modelParameters.tree_sum_length = 0; % 0, 1, 2
        modelParameters.tree_binsize = 0; % 40, 80, 120
        modelParameters.tree_freq = 0;
        modelParameters.start_floor_binsize = floor(300 / modelParameters.tree_binsize);
        modelParameters.discrimType = 'quadratic'; %['linear' 'quadratic' 'pseudoquadratic']
        modelParameters.pc_components = 40;
        modelParameters.time_norm = false;
        modelParameters.mean_adjust = true;
        modelParameters.max_norm = true;  
        modelParameters.x_start_path = true;
        modelParameters.use_true = false;
        
    elseif modelParameters.model_name == "Avg. Traj. w/ Convergence"
        modelParameters.init_idx = 320;
        modelParameters.init_bag_size = 1;%4;
        modelParameters.mid_idx = 360;
        modelParameters.mid_bag_size = 1;%8;
        modelParameters.bag_data_split = 0.7;
        modelParameters.w = 0.9;
        modelParameters.hold_out = 0.5;
        modelParameters.validateQD = false;
        modelParameters.tree_filter_length = 0; % 0, 1, 2
        modelParameters.tree_sum_length = 0; % 0, 1, 2
        modelParameters.tree_binsize = 0; % 40, 80, 120
        modelParameters.tree_freq = 0;
        modelParameters.start_floor_binsize = floor(300 / modelParameters.tree_binsize);
        modelParameters.discrimType = 'quadratic'; %['linear' 'quadratic' 'pseudoquadratic']
        modelParameters.pc_components = 40;
        modelParameters.time_norm = false;
        modelParameters.mean_adjust = true;
        modelParameters.max_norm = true;  
        modelParameters.x_start_path = false;
        modelParameters.use_true = true;
        
    elseif modelParameters.model_name == "Avg. Traj."
        modelParameters.init_idx = 320;
        modelParameters.init_bag_size = 1;%4;
        modelParameters.mid_idx = 360;
        modelParameters.mid_bag_size = 1;%8;
        modelParameters.bag_data_split = 0.7;
        modelParameters.w = 0.0;
        modelParameters.hold_out = 0.5;
        modelParameters.validateQD = false;
        modelParameters.tree_filter_length = 0; % 0, 1, 2
        modelParameters.tree_sum_length = 0; % 0, 1, 2
        modelParameters.tree_binsize = 0; % 40, 80, 120
        modelParameters.tree_freq = 0;
        modelParameters.start_floor_binsize = floor(300 / modelParameters.tree_binsize);
        modelParameters.discrimType = 'quadratic'; %['linear' 'quadratic' 'pseudoquadratic']
        modelParameters.pc_components = 40;
        modelParameters.time_norm = false;
        modelParameters.mean_adjust = true;
        modelParameters.max_norm = true;  
        modelParameters.x_start_path = false;
        modelParameters.use_true = true;

    elseif modelParameters.model_name == "QDA w/ Avg. Traj. w/ Convergence w/ Bagging"
        modelParameters.init_idx = 320;
        modelParameters.init_bag_size = 8;
        modelParameters.mid_idx = 360;
        modelParameters.mid_bag_size = 4;
        modelParameters.bag_data_split = 0.99;
        modelParameters.w = 0.9;
        modelParameters.hold_out = 0.8;
        modelParameters.validateQD = false;
        modelParameters.tree_filter_length = 0; % 0, 1, 2
        modelParameters.tree_sum_length = 0; % 0, 1, 2
        modelParameters.tree_binsize = 0; % 40, 80, 120
        modelParameters.tree_freq = 0; % 160, 320
        modelParameters.start_floor_binsize = floor(300 / modelParameters.tree_binsize);
        modelParameters.discrimType = 'quadratic'; %['linear' 'quadratic' 'pseudoquadratic']
        modelParameters.pc_components = 10;
        modelParameters.time_norm = false;
        modelParameters.mean_adjust = true;
        modelParameters.max_norm = true;
        modelParameters.use_average_traj = false;
        modelParameters.use_true = false;

    elseif modelParameters.model_name == "QDA w/ Avg. Traj. w/ Convergence w/ Ensemble Tree"
        modelParameters.init_idx = 320;
        modelParameters.init_bag_size = 8;
        modelParameters.mid_idx = 360;
        modelParameters.mid_bag_size = 4;
        modelParameters.bag_data_split = 0.99;
        modelParameters.w = 0.9;
        modelParameters.hold_out = 0.8;
        modelParameters.validateQD = false;
        modelParameters.tree_filter_length = 2; % 0, 1, 2
        modelParameters.tree_sum_length = 2; % 0, 1, 2
        modelParameters.tree_binsize = 40; % 40, 80, 120
        modelParameters.tree_freq = 320; % 160, 320
        modelParameters.start_floor_binsize = floor(300 / modelParameters.tree_binsize);
        modelParameters.discrimType = 'quadratic'; %['linear' 'quadratic' 'pseudoquadratic']
        modelParameters.pc_components = 10;
        modelParameters.time_norm = false;
        modelParameters.mean_adjust = true;
        modelParameters.max_norm = true;
        modelParameters.use_average_traj = false;
        modelParameters.use_true = false;

    end
end