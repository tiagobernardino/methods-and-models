%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stochastic neoclassical growth model (using VFI with grid search)
% based on Pedro Brinca's code
% Tiago Bernardino, IIES
% August, 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc

%% Model Parameters
alpha   = 0.36;   %capital share
beta    = 0.96;   %discount factor
delta   = 0.06;   %depreciation rate
sigma   = 2;      %coefficient of risk aversion
mu      = 0;      %constant of the AR(1) process
rho     = 0.9;    %persistency of the income shock
std_z   = 0.05;   %std dev of the income shock

%% Computational Parameters
nk      = 1000;   %#k grid points
ns      = 2;      %#possible states in the economy

tolv    = 10^-5;  %tolerance level VF
maxiter = 3000;   %maximum #iterations

NL_grid = 1;      %Non-linear grid? (Yes = 1, No = 0)

%% Discretizing the z grid
w = 0.5 + rho/4;
sigmaZ = std_z/sqrt(1-rho^2);
baseSig = w*std_z +(1-w)*sigmaZ;
[S,S_prob] = tauchenhussey(ns,mu,rho,std_z,baseSig);

z = exp(S)/mean(exp(S));  %center around 1
z_prob = S_prob;

clear w sigmaZ baseSig S S_prob

%% Steady state values
z_ss  = mean(z);
k_ss  = ((alpha*beta*z_ss)/(1-beta*(1-delta)))^(1/(1-alpha));
y_ss  = ((alpha*beta*z_ss)/(1-beta*(1-delta)))^(alpha/(1-alpha));
c_ss  = ((alpha*beta*z_ss)/(1-beta*(1-delta)))^(1/(1-alpha))*...
       (1-beta*(1-delta)-alpha*beta*delta)/(alpha*beta*z_ss);

%% Discretizing the capital grid
kmin    = 0.75*k_ss;
kmax    = 1.5*k_ss;

if NL_grid == 1  %non-linear grid
    K = linspace(0,log(kmax-kmin+1),nk);
    K = exp(K)-1+kmin;
else             %linear grid
    K = linspace(kmin,kmax,nk);
end

%% Value function iteration
%Initial values
V    = ones(ns,1)*(K.^(1-sigma)/(1-sigma));
g    = zeros(ns,nk);

%Initialize the loop
iter = 0;          %initialize iteration counter VF
diffv = tolv+eps;  %criterion for convergence

%Grid search
tStart_v = tic;
while diffv > tolv && iter < maxiter
    iter = iter+1; %iteration counter
    newV = zeros(ns,nk) * NaN;
    
    %Maximization
    for iz=1:ns %for each z
        for ik=1:nk %for each k
            cons = z(iz)*K(ik)^alpha+(1-delta)*K(ik)-K;  %get cons from the BC given state (z,k)
            cons = max(cons,1e-8);  %no negative consumption
            v = (cons.^(1-sigma))/(1-sigma) + ...
                beta*z_prob(iz,:)*V;  % at state (z,k) get v for all k'
            [newV(iz,ik),index] = max(v);  %get the maximum
            g(iz,ik) = K(index);  %get the policy function
        end
    end
    
    %Check VF convergence
    diffv = norm(newV-V);
    if rem(iter,10) == 0  %show convergence pattern every 10 iterations
        disp(['VF iter = ', num2str(iter), '. Norm= ',num2str(diffv)])
    end
    
    %Update new VF
    V = newV;
end
tEnd_v = toc(tStart_v);

%Print infos about VFI
if iter < maxiter
    disp(['Value function converged in ', num2str(iter), ' iterations :) Time taken: ',num2str(tEnd_v)])
else
    disp(['Value function did not converge :( Time taken: ',num2str(tEnd_v) ])
end


%% Plot Value Function and Policy Function
figure(1) %Value function
plot(K,V,'LineWidth',2)
title('Value Function')
xlabel('k')
ylabel('V')
legend('z_{low}','z_{high}','Location','southeast')

figure(2) %Savings Policy Function
plot(K,g,'LineWidth',1)
title('Policy Functions')
xlabel('k')
ylabel('k')
hline = refline(1,0);
hline.Color = 'black';
hline.LineStyle = '--';
legend('z_{low}','z_{high}','45 line','Location','southeast')


%% Simulation
%Set-up
rng(750);     %random number seed
Tsimul = 1000;  %#periods to simulate

%Simulate z
zsim     = ones(Tsimul+1,2)*NaN;  %storage values of [i,z(i)]
[~, col] = size(z);
zsim(1,1) = col; %first state
for i = 2:Tsimul+1
  this_step_dist = z_prob(zsim(i-1,1),:); %Prob at state
  cumulative_distribution = cumsum(this_step_dist);
  random = rand(); %generate a random number in [0,1]
  zsim(i,1) = find(cumulative_distribution>random,1); %get the first to be higher in the cdf
end
zsim(:,2) = z(zsim(:,1)); %convert state number to value of z

%Simulate capital
ksim     = zeros(Tsimul+1,1)*NaN; %storage values of k
[~,iss]  = min(abs(k_ss-K)); %start in the SS value of capital (find the SS index)
ksim(1)  = K(iss);
%Note that g has the policy function for all possible values of (z,k)
for j=2:Tsimul+1 %simulate k
   ksim(j)   = g(find(z==zsim(j,2)),find(K==ksim(j-1)));
end
 
%Simulate output and consumption
ysim = zsim(1:end-1,2).*ksim(1:end-1).^alpha; %simulate y
csim = ysim+(1-delta)*ksim(1:end-1)-ksim(2:end); %simulate c

%% Simulation Plots
figure(3)
subplot(221)
plot(zsim(1:end-1,2),'LineWidth',1,'Color','black')
hold on
yline(z_ss,'--')
title('Income shock')
xlabel('time')
ylabel('z')

subplot(222)
plot(csim,'LineWidth',1,'Color','black')
hold on
yline(c_ss,'--')
title('Consumption')
xlabel('time')
ylabel('c')

subplot(223)
plot(ysim,'LineWidth',1,'Color','black')
hold on
yline(y_ss,'--')
title('Output')
xlabel('time')
ylabel('y')

subplot(224)
plot(ksim(1:end-1),'LineWidth',1,'Color','black')
hold on
yline(k_ss,'--')
title('Capital')
xlabel('time')
ylabel('k')