function QoI_LSTMallPlot(PQoIinpAll,predPQoIAll,rmsePQoIAll,UserNO)
%this plots the prediction for all the Professors 
% from the output of:
% [PQoIinp,predPQoI,rmsePQoI] = QoI_LSTMall(P_QoI);

[r,c]=size(predPQoIAll);% to find the size of the prediction (columns)
[r1,c1]=size(PQoIinp);
figure(1)
plot(PQoIinp(1:c-1))
hold on

idx = c1:(c1+c);
plot(idx,[PQoIinp(c1) predPQoI],'.-')
hold off

    xlabel("Days")

ylabel("QoI")
title("QoI Observed and Forecast values")
legend(["Observed" "Forecast"])
% Compare the forecasted values with the test data.
if 0

figure(2)
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("QoI")
title("QoI Forecast Values")

subplot(2,1,2)
stem(YPred - YTest)

    xlabel("Days")


ylabel("Error")
title("RMSE = " + rmse)

end




end