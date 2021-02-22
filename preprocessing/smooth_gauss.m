function filtered_data = smooth_gauss(input_data, L)
% Function that applies Gaussian smoothing with the window of width L
% across each individual row of the input_data matrix

% Input_data - data of neural spikes
% L - width of gaussian window

% Assumes the following data structure:
% - Row number - neuron number
% - Column number - time

    w = gausswin(L);
    filtered_data = filter(w,1,input_data, [], 2);
    
end

