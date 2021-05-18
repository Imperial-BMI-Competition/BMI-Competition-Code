function [data] = tree_apply_filter(data, x_filter, filter_length)
    if filter_length > 1
       for row = 1:size(data, 1)
           x = data(row,:);
           data(row,:) = [zeros(1, filter_length-1) conv(x, x_filter/filter_length,'valid')];
       end
    end
end