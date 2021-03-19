function [decoder] = SDA_decoder(data)
    % Estiamtes Neurons Tuning curve for population vector decoding

    fsamp = 1000;
    angles = [30    70   110   150   190   230     310   350];
    T_test = 320;% test on predicitng angle given the initial 320 msec only 

    %Get the dishcarge rates of all  neurons, angles and tasks
    for row = 1:size(data,1)
        for angle = 1:size(data,2)

            trial = data(row,angle).spikes;
            n_spikes(:,angle,row) = sum(trial(:,1:end),2);
            n_spikes_test(:,angle,row) = sum(trial(:,1:T_test),2);

        end
    end
    
    % Check of neurons are directionally tuned 
    theta_radians = deg2rad(angles);
%     [x, y] = pol2cart(theta_radians, 1);  
%     unit_vectors = [x;y];
%     [r_max,s_a] = max(firing_rate,[],2);
 
     % Build Fetures Vector 
    F = []; F_ = [];
    y_true = []; y_true_ = [];

    F_test = [];
    y_true_test = [];

    for i = 1: size(n_spikes,2)
        for j = 1:  size(n_spikes,3)
            total_n_spikes = sum(n_spikes(:,i,j));
            total_n_spikes_test = sum(n_spikes_test(:,i,j));

            f = n_spikes(:,i,j)';
            f_test = n_spikes_test(:,i,j)';
            
            % Training 
            f = f./total_n_spikes;
            F = [F;f]; 
            y_true = [y_true; i];

            % Testing - 320 msec only
            f_test = f_test./total_n_spikes_test;
            F_test = [F_test;f_test]; 
            y_true_test = [y_true_test; i];
            
        end
    end
    % Train classifier
    cv = cvpartition(size(F_test,1),'HoldOut',0.5);
    idx = cv.test;
    F_all = [[F, y_true];[F_test(idx,:),y_true_test(idx,:)]];
    [angle_classifier, validationAccuracy] = trainClassifierQDA(F_all);


%     fprintf('\nReaching angle decoder (Quadratic Discriminant) Validation Accuracy %2.1f %%',100*validationAccuracy);
    decoder = {};
    
    % Format Output
    decoder.model = 'Quadratic Discriminant';
    decoder.classifier = angle_classifier;
    decoder.accuracy =  validationAccuracy;

end
