function out = get_diffs(pos_trajectory, difference)
     [n,m] = size(pos_trajectory);
     shifted = NaN(n,m);
     shifted(:,difference+1:m) = pos_trajectory(:,1:m-difference);
     out = pos_trajectory - shifted;
     out(isnan(out))=0;
end