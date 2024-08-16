%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stochastic neoclassical growth model (using log-linear method)
% based on Pedro Brinca's code
% Tiago Bernardino, IIES
% August, 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc

%% Model Parameters
alpha   = 0.36;   %capital share
beta    = 0.96;   %discount factor
delta   = 0.1;    %depreciation rate
sigma   = 2;      %coefficient of risk aversion
mu      = 0;      %constant of the AR(1) process
rho     = 0.9;    %persistency of the income shock
std_z   = 0.05;   %std dev of the income shock

%% Steady state values
z_ss = 1;
k_ss  =((alpha*beta*z_ss)/(1-beta*(1-delta)))^(1/(1-alpha));
y_ss  =((alpha*beta*z_ss)/(1-beta*(1-delta)))^(alpha/(1-alpha));
c_ss  =((alpha*beta*z_ss)/(1-beta*(1-delta)))^(1/(1-alpha))*(1-beta*(1-delta)-...
alpha*beta*delta)/(alpha*beta*z_ss);

%% Compute numerical derivatives with perturbation
P       = [alpha beta delta sigma]; %vector of paramenters
Z       = [log(k_ss);log(k_ss);log(k_ss);log(z_ss);log(z_ss)];
epsilon = max(abs(Z)*1e-5,1e-8); %perturbation

for i=1:5 %derivative wrt the 5 inputs
  Zp       = Z;
  Zm       = Z;
  Zp(i)    = Z(i)+epsilon(i);
  Zm(i)    = Z(i)-epsilon(i);
  dR(i,1)  = (residual(Zp,P)-residual(Zm,P))/(2*epsilon(i)); %call residual.m
end

%% Solve for gamma_k and gamma_z
a2 = dR(1); %assign value
a1 = dR(2);
a0 = dR(3);
b1 = dR(4);
b0 = dR(5);
gamma_k = roots([a2 a1 a0]);
gamma_k = gamma_k(2); %choose the one within the unit-root 
gamma_z = -(b0 + b1*rho)/(a1+a2*gamma_k + a2);

%% Simulation
rng(75);
simul    = 1000;             %#periods to simulate

% Simulate z (AR1 process)
zsim    = ones(simul+1,1);  %storage value of z
zsim(1) = 0;  %first state
for i = 1:simul
  zsim(i+1) = mu + rho*zsim(i) + normrnd(0,std_z);
end
zsim = exp(zsim);

% Simulate capital (based on the policy function)
ksim = zeros(simul+1,1); %storage values of k
for j = 1:simul
   ksim(j+1) = gamma_k * ksim(j) + gamma_z * log(zsim(j));
end
ksim = exp(ksim);

%Simulate output and consumption 
ysim = zsim(1:end-1).*ksim(1:end-1).^alpha;      %simulate y
csim = ysim+(1-delta)*ksim(1:end-1)-ksim(2:end); %simulate c

ysim = ysim*y_ss;
csim = csim*c_ss;
ksim = ksim*k_ss;

%% Plots
figure(3)
subplot(221)
plot(zsim(1:end-1),'LineWidth',1.5,'Color','black')
hold on
yline(z_ss,'--')
title('Income shock')
xlabel('time')
ylabel('z')

subplot(222)
plot(csim,'LineWidth',1.5,'Color','black')
hold on
yline(c_ss,'--')
title('Consumption')
xlabel('time')
ylabel('c')

subplot(223)
plot(ysim,'LineWidth',1.5,'Color','black')
hold on
yline(y_ss,'--')
title('Output')
xlabel('time')
ylabel('y')

subplot(224)
plot(ksim(1:end-1),'LineWidth',1.5,'Color','black')
hold on
yline(k_ss,'--')
title('Capital')
xlabel('time')
ylabel('k')
%print(gcf, '-depsc', '..\book\Figures\neoclassical_simul_loglin.eps');
