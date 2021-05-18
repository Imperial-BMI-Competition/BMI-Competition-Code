% Test Script to give to the students, March 2015
% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

function [RMSE, acc, time] = testFunction_for_students_MTb(seed, modelParameters)

% clc; %close("all");

tic;

load ../monkeydata0.mat

% Set random number generator
rng(seed);
ix = randperm(length(trial));

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

meanSqError = 0;
n_predictions = 0;  

% figure
% hold on
% axis square
% grid

% Train Model
modelParameters = positionEstimatorTraining(trainingData, modelParameters);

attempts = 0;
correct = 0;

for tr=1:size(testData,1)
%     display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
%     pause(0.001)
    for direc=randperm(8)
        modelParameters.true_dir = direc;
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            %current = testData(tr,direc).handPos(1:2,320:20:t);
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
        end
        attempts = attempts + 1;
        if direc == modelParameters.estimated_angle
            correct = correct + 1;
        end
        n_predictions = n_predictions+length(times);
    end
end

% legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions);
acc = correct / attempts;
fprintf('\nRMSE %2.3f \n',RMSE);
fprintf('Accuracy %2.1f %%\n',100*acc);
time = toc;
fprintf('Time %2.4f \n',time);
fprintf('---------\n');

end
