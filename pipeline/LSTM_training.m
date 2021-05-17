function [model_minibatch16_neural] = LSTM_training(trials)
    
    angles = [30 70 110 150 190 230 310 350];
    
    [x_angles, y_angles] = pol2cart(angles.*(pi/180), 1);  
    n_trials = length(trials); %number of trials
    dataTrain = {}; %needs to be in cell array for subsequent code


    count = 1;
    for i = 1:n_trials %number of trials
        for j = 1:length(angles)     %number of types of directions (experiments)
            N = trials(i,j).handPos(1:2,:); %select only x and y, not z
            for T = 1 : size(N,2)
                dataTrain{count} = [N(:,T); x_angles(j); y_angles(j)]; % make one long row vector with features of position
                count = count + 1;
            end
        end
    end

    %Standardise all the features by mean of that feature across all trials
    dataTrain2 = cell2mat(dataTrain);  %to calculate mean and standard deviation of each feature to normalise
    train_mu = mean(dataTrain2, 2);
    train_sig = std(dataTrain2, [], 2); 

    dataTrain_standardisedX = {};
    dataTrain_standardisedY = {};
    
   
    for uu = 1:size(dataTrain,2) - 20
        trial = dataTrain{uu};
        trial = (trial - train_mu) ./ train_sig;
        
        trial_next = (dataTrain{uu+20} - train_mu) ./ train_sig;
        
        dataTrain_standardisedX{uu} = trial;
        dataTrain_standardisedY{uu} = trial_next(1:2);

    end

    XTrain = dataTrain_standardisedX;  %to predict use all data but end time step
    YTrain = dataTrain_standardisedY; 

    disp("Created test dataset")


    % Initialise network
    % From https://www.mathworks.com/help/deeplearning/ug/long-short-term-memory-networks.html
    numHiddenUnits = 100;
    numResponses = size(YTrain{1},1);
    numFeatures = size(XTrain{1},1); %x, y and z. later we will change this to be more than just hand position


    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits,'OutputMode','sequence') %,'OutputMode','last') %for a single prediction output (not a sequence)
        fullyConnectedLayer(numResponses)
        regressionLayer];

    % From https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html
    options = trainingOptions('adam', ...
        'MaxEpochs',2, ...
        'MiniBatchSize',16, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...  %after 125 epochs drop learning rate by factor of 0.2
        'LearnRateDropFactor',0.2, ...
        'Verbose',0, ...
        'Plots','training-progress');

    disp("Initialised Network")

    % Train Network
    net = trainNetwork(XTrain,YTrain,layers,options);

    % Test Network:
     net = predictAndUpdateState(net,XTrain, 'MiniBatchSize', 1);  %initialise network

    modelParameters.net = net;
    modelParameters.mu = train_mu;
    modelParameters.sig = train_sig;

    model_minibatch16_neural = modelParameters;
    %save model_minibatch16_lr0.01_epoch10_neural
end
