%% This file will swipe the noise level
clc;clear all;close all;
%% First load the noise and simulation data
load('GeneratedNoiseV2.mat')
addpath("Result")
addpath("Functions")
%% Define the parameters for the WeakSINDy_AG
% Library order
polyorder=2;
usesine=0;

% Optimization parameter
gamma=0;
tau = 1;

% Number of test function
K = 1000;

% Test function order
poly = 2; 

% Test function support size
s = 16;
r_whm = 30;

% Parameters for the WeakSINDy
wsindy_params = {s, K, poly, tau};

% Define the thresholding parameters
lam=linspace(0.01,0.95,30);
lamNum=length(lam);

% Define a matrix to store the values
Theta_test=poolData(x_test,stateVar,polyorder,usesine);
Xi0=zeros(lamNum,size(Theta_test,2),stateVar);
TestingScore=zeros(lamNum,1);
SuccessOrNot_WS=zeros(N_run,NoiseNum);
Xi_WS=zeros(lamNum,size(Theta_test,2),stateVar);

% Store the parameter error
Eparm_WS=zeros(N_run,NoiseNum);

%% Next run a for loop to test each noise level
pin=0;

for i=1:N_run
    fprintf(strcat("This is the run:",string(i),"\n"))
    for j=1:NoiseNum
        fprintf(strcat("\t","Using noise level as: ",string(NoisePercentageToSwipe(j)),"\n"))
        pin=pin+1;
        % Get the observation data
        xn=squeeze(xn_List(pin,:,:));
        Theta=poolData(xn,stateVar,polyorder,usesine);
        
        % Swipe the thresholding parameter
        parfor k=1:lamNum
            fprintf(strcat("\t\t","Swiping lambda, using lambda as: ",string(lam(k)),"\n"))
            
            % Get the result for current lam
            Xi0(k,:,:)=WeakSINDy_AG(t,xn,Theta,stateVar,lam(k),gamma,r_whm,wsindy_params);
            
            % Calculate the error on the test data
            TestingScore(k,1)=norm(dx_test-Theta_test*squeeze(Xi0(k,:,:)))/norm(dx_test);
            
        end
        
        % Test whether the structure of the model is correct
        for k=1:lamNum
            if norm((Xi_base~=0)-(squeeze(Xi0(k,:,:))~=0))==0
                SuccessOrNot_WS(i,j)=1;
                Xi_dummy=squeeze(Xi0(k,:,:));
                Xi_WS(pin,:,:)=Xi_dummy;
            end
        end
        
        % If no model has the correct structure, choose the model that has
        % lowest testing error as final model
        if SuccessOrNot_WS(i,j)==0
           % Now select the best Xi
            [minVal,index]=min(TestingScore);
        
            % Store the current result
            Xi_dummy=squeeze(Xi0(index,:,:));
            Xi_WS(pin,:,:)=Xi_dummy;
        end
        
        % Store the parameter error
        Eparm_WS(i,j)=norm(Xi_base-Xi_dummy)/norm(Xi_base);
        
    end
end

%% Save the results
Ep_WS_001=Eparm_WS;
SuccessOrNot_WS_001=SuccessOrNot_WS;

save ("Result\W_SINDy_dt_001.mat","Ep_WS_001","SuccessOrNot_WS_001","NoiseNum","NoisePercentageToSwipe")
