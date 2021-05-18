# BMI-Competition-Code

 - Kalman_QDA_final
 - LSTM_network
 - Spectrogram_CNN
 - submission_1
 - submission_2
 - submission_3
 - submission_4
 - submission_5
 - submission_6
 - submission_7 
 - submission_8 
 - submission_best_model: QDA/LDA with Converged (Includes bagging, and ensemble trees)
 - submission_kalman

Reproducing results:
1) To reproduce the results from the best model, go into the folder submission_best_model.
2) Inside the get_params.m file you can see the different model configurations and the string to call that config. Select the string of the model you wish to select and set line 2 in run_with_defined_model.m as:
  - modelParameters.model_name = "your-selected-name";
3) call the script run_with_defined_model.m

Extra - Reproducing Figures:
1) Population Vector: popVectorFig.m
1) QDA: QDA_decoder.m
