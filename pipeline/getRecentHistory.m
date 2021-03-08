function recent_history = getRecentHistory(pos_trajectory, lookback, difference)
    pos_trajectory = fliplr(pos_trajectory);
    
    out = pos_trajectory(:, 1:difference:end);
    pos_size = size(out);
    
    start_idx = pos_size(2)-lookback;
    if start_idx <= 0
        recent_history = zeros(2, lookback);
        if pos_size(2) > 0
            recent_history(:, 1:pos_size(2)) = out;
        end
    else
        recent_history = out(:, 1:lookback);
    end
    recent_history = fliplr(recent_history);
end
