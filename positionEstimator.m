
function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  
  
trials = test_data;
trials.handPos = [trials.startHandPos, trials.decodedHandPos];
size_t = length(trials); %number of trials
t = size(trials.spikes);  %split spikes into 20 sec windows and take average over them
spikes = [];
for i = 300:20:t(2)-20
    s = mean(trials.spikes(:,i:i+20), 2); %take mean along the rows
    spikes = [spikes, s];
end
dataTest = [trials.handPos; spikes]; %use neural spikes as feature as well as position

%Standardise all the features by save values as training set
test_mu = modelParameters.mu;
test_sig = modelParameters.sig;

dataTest_standardised = (dataTest - test_mu)./ (test_sig +0.0000001); %so it never divides by 0

XTest = dataTest_standardised;
YPred = XTest; %initialise as the same

net = modelParameters.net;

s = size(XTest);
numTimeStepsTest = s(2);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i-1),'ExecutionEnvironment','cpu');
end
newModelParameters.net = net;
newModelParameters.mu = test_mu;
newModelParameters.sig = test_sig;

% Unnormalise predictions
YPred_units = test_sig.*YPred + test_mu;

% for t = 1:length(YPred) % number of test trials
%     y = YPred(t);
%     y = test_sig.*y + test_mu;
%     YPred_units= [YPred_units; y];    
% end

predict = YPred_units(:,end);
x = predict(1);
y = predict(2);
   
end