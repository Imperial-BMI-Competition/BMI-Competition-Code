function x_filter = tree_get_filter(filter_length)
    if filter_length > 1
        x_filter = ones(1, filter_length);
        for i = 2:filter_length
            x_filter(i) = x_filter(i) / (i * 0.5); 
        end
        x_filter = x_filter - x_filter(end);
    else
        x_filter = 0;
    end
end

