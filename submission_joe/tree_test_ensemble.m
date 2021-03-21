clc; clear; close("all");

load ../monkeydata0.mat

rng(2013);
ix = randperm(length(trial));

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

% send in batches of data and predict the class

model = tree_ensemble(trainingData, 3, 4, 100);
