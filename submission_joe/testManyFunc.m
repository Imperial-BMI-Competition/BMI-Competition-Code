function [rmses] = testManyFunc(seeds)
    rmses = [];
    for i = 1:size(seeds,2)
        clearvars -except seeds i rmses
        rmse = testFunction_for_students_MTb(seeds(i));
        rmses = [rmses rmse];
    end
    m_rmse = mean(rmses)
end