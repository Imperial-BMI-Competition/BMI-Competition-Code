
function [x, y, newModelParameters] = positionEstimator(trials, modelParameters, direc)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(trials, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - trials:
  %     trials(m).trialID
  %         unique trial ID
  %     trials(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     trials(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     trials(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             trials.trialID = 1;
  %             trials.startHandPos = [0; 0]
  %             trials.decodedHandPos = []
  %             trials.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             trials.trialID = 1;
  %             trials.startHandPos = [0; 0]
  %             trials.decodedHandPos = [2.3; 1.5]
  %             trials.spikes = 98x340 matrix of spiking activity
  
 
trials = trials;
trials.handPos = [trials.startHandPos, trials.decodedHandPos];
size_t = length(trials); %number of trials
t = size(trials.spikes);  %split spikes into 20 sec windows and take average over them

spikes = []; %first time block
dpos = [] ; 
diff = trials.startHandPos(:,1);       %first time block

for i = 1:20:320-20
    s = mean(trials.spikes(:,i:i+19), 2); %take mean along the rows
    spikes = [spikes, s];
    dpos = [dpos,diff];
    diff = [0.0;0]; %since no measurements here

end

count = 1;
if t(2) > 320
    for i = 320:20:t(2)-20
        s = mean(trials.spikes(:,i:i+20), 2); %take mean along the rows
        spikes = [spikes, s];
        diff = trials.handPos(:,count) - trials.handPos(:,count+1);
        dpos = [dpos, diff];

        count = count +1;
    end
end

%e = get_experiment or get angle, that Manfredi is developing
e = direc;
experiment = ones(1,length(dpos))* e;
dataTest = [dpos; experiment; spikes]; %use neural spikes as feature as well as position

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

%We predict CHANGE in position, now change this to absolute position:
predict = YPred_units(:,end);

x = trials.handPos(1,end) + predict(1);
y = trials.handPos(2,end) + predict(2);

   
end