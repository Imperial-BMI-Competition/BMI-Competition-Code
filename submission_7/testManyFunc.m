function [rmses, m_rmse] = testManyFunc(seeds, modelParameters)
    rmses = [];
    for i = 1:size(seeds,2)
        clearvars -except seeds i rmses modelParameters mp
        rmse = testFunction_for_students_MTb(seeds(i), modelParameters);
        rmses = [rmses rmse];
    end
    m_rmse = mean(rmses);
end
