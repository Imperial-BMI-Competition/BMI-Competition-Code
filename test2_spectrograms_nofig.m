YPredicted = predict(net,XValidation);

predictionError = YValidation - YPredicted;

squares = predictionError.^2;
rmse = sqrt(mean(squares))
