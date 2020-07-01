function QoI_LSTMall2_par(QoIAlldata,StartEnd,flag1)
%this claculates the prediction for all the Professors 
% calls QoI_LSTM.m with interpolation (days display) and no plots
% if flag1=1 Profs, else Students
% QoI_LSTMall(P_QoI,1) for Profs and QoI_LSTMall(P_QoI,2)for Students

[r,c]=size(QoIAlldata); % r: number of profs, c: number of weeks/days (for interpolation)
if flag1==1
%h=waitbar(0,'Please wait...Ongoing Prediction for Professors QoI');
filename=['ResultsPQoI' num2str(StartEnd(1)) '-' num2str(StartEnd(2)) '.mat'];
for i=1:r %no of Profs
 StartEnd=StartEnd
 i=i %to diplay the current prof
       [DD,pDD,errDD] = QoI_LSTM(QoIAlldata(i,:),1,0); % per days, no plot
       PQoIinp(i,:)=DD;
       predPQoI(i,:)=pDD;
       rmsePQoI(i)=errDD;
    %  waitbar(i/r,h)
       save(filename,'PQoIinp','predPQoI','rmsePQoI','i');
end
% close h
else
    filename=['ResultsSQoI' num2str(StartEnd(1)) '-' num2str(StartEnd(2)) '.mat'];
%     h=waitbar(0,'Please wait...Ongoing Prediction for Students QoI');

xIN1=interp1(1:c,QoIAlldata(1,:),1:7/(length(QoIAlldata(1,:))-1):length(QoIAlldata(1,:)));
    NN=length(xIN1);
 SQoIinp=zeros(r,NN);
 predSQoI=zeros(r, NN-floor(0.90*NN)-1);
 rmseSQoI=zeros(1,r);
 parpool('local',24);
parfor ii=1:r %no of Students
tic
%  StartEnd(2)+ii-1 %to diplay the current student
  com=['Range: ' num2str(StartEnd(1)) '-' num2str(StartEnd(2)) '; iterON=' num2str(ii)];
  fprintf('%s\n',com)
  
       [DD,pDD,errDD] = QoI_LSTM(QoIAlldata(ii,:),1,0); % per days, no plot
       SQoIinp(ii,:)=DD;
       predSQoI(ii,:)=pDD;
       rmseSQoI(ii)=errDD;
%        waitbar(i/r,h)
t2=toc;
 com2=['Range: ' num2str(StartEnd(1)) '-' num2str(StartEnd(2)) '; iterOFF=' num2str(ii) 'elapsed time=' num2str(t2)];
  fprintf('%s\n',com2)
end
       save(filename,'SQoIinp','predSQoI','rmseSQoI');
% close h
end
