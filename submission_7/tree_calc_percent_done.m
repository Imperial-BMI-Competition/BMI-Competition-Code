function [warmup_percent_done, traj_percent_done] = tree_calc_percent_done(idx, start_floor_binsize, l)
    if idx >= start_floor_binsize
         traj_percent_done = 1 - ((l - idx) / (l - start_floor_binsize));
         warmup_percent_done = 1.0;
    else
         traj_percent_done = 0.0;
         warmup_percent_done = 1 - ((start_floor_binsize - idx)/ start_floor_binsize);
    end
end