# BMI-Competition-Code
The folder called 'submission' contain the decoder submitted during the BMI competition. 
 - Kalman_QDA_final
 - LSTM_network
 - Spectrogram_CNN
 - submission_1: Population Vector with avg. trajectory
 - submission_2: Ensemble discriminant analysis  with avg. trajectory
 - submission_3: Ensemble discriminant analysis  with avg. trajectory
 - submission_4: QDA with avg. trajectory
 - submission_5: QDA with avg. trajectory (parameters improvements)
 - submission_6: QDA with avg. trajectory (parameters improvements)
 - submission_7: QDA with avg. trajectory (parameters improvements)
 - submission_8: QDA with avg. trajectory (parameters improvements)
 - submission_best_model: QDA/LDA with Converged (Includes randoms search, bagging, and ensemble trees)
 - submission_kalman

Reproducing results:
1) To reproduce the results from the best model, go into the folder submission_best_model.
2) Inside the get_params.m file you can see the different model configurations and the string to call that config. Select the string of the model you wish to select and set line 2 in run_with_defined_model.m as:
  - modelParameters.model_name = "your-selected-name";
3) call the script run_with_defined_model.m

Extra - Reproducing Figures:
1) Population Vector: popVectorFig.m
2) QDA: QDA_decoder.m
