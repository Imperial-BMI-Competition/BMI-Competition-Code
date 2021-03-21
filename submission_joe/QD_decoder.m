function [decoder] = QD_decoder(F_all, validateQD)
    % Estiamtes Neurons Tuning curve for population vector decoding
    [angle_classifier, validationAccuracy] = trainClassifierQDA(F_all, validateQD);
    decoder = {};
    decoder.model = 'Quadratic Discriminant';
    decoder.classifier = angle_classifier;
    decoder.accuracy =  validationAccuracy;
end
