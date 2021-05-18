function [rmses, m_rmse] = testManyFunc(seeds, modelParameters)
    rmses = [];
    accs = [];
    times = [];
    for i = 1:size(seeds,2)
        clearvars -except seeds i rmses modelParameters mp accs times
        current_seed = seeds(i);
        [rmse, acc, time] = testFunction_for_students_MTb(current_seed, modelParameters);
        rmses = [rmses rmse];
        accs = [accs acc];
        times = [times time];
    end
    m_rmse = mean(rmses);
    
    fprintf("===================================\n");
    fprintf('\nRMSE %2.3f \n', mean(rmses));
    fprintf('Accuracy %2.1f %%\n',100*mean(accs));
    fprintf('Time %2.4f \n',mean(times));
    fprintf("===================================\n");
end
