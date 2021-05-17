function [decoder] = QD_decoder(F_all, validateQD, discrimType, pc_components)
    % Estiamtes Neurons Tuning curve for population vector decoding
    [angle_classifier, validationAccuracy] = trainClassifierQDA(F_all, validateQD, discrimType, pc_components);
    decoder = {};
    decoder.model = 'Quadratic Discriminant';
    decoder.classifier = angle_classifier;
    decoder.accuracy =  validationAccuracy;
end
