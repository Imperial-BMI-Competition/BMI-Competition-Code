function raster_plot(data)

% Plot Params 
spike = data.trial;
plotwidth=1;     % spike thickness
colors = jet(size(spike,1));
plotcolor='k';   % spike color
gap=1.5;    % distance between trials
fs = 1000;  %  sampling rate
direction = 1;
neuron = 1;
figure;
for trial_ = 1 : size(spike,1)
    train = data.trial(trial_,direction).spikes(neuron,:);
    times = find(train == 1);
    y = (trial_ * gap + 1).* ones(size(times,2),1);
    
    scatter(1000.*(times./fs), y,[],'|','MarkerEdgeColor',colors(trial_,:),...
        'LineWidth',plotwidth); hold on;
end

xlabel('Time (msec)','Fontsize',14);
ylabel('Trial','Fontsize',14);
title('Raster plot','Fontsize',14);
end