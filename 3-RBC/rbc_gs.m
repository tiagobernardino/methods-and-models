%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Real Business Cycle model (using VFI with golden rule search)
% Tiago Bernardino, IIES - Stockholm University
% August, 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc

%% Model Parameters
alpha   = 0.36;    %capital share
beta    = 0.96;    %discount factor
delta   = 0.06;    %depreciation rate
gamma_n = 0.00;    %population growth
gamma_z = 0.00;    %tech growth
rho     = 0.9;     %productivity shock persistence parameter
std_z   = 0.05;    %std dev of the income shock
mu      = 0;       %constant of the AR(1) process

%% Computational Parameters
nk = 10;           %grid points of k
ns = 2;            %number of possible stochastic states

tolv    = 10^-5;   %tolerance lvl for VF convergence
tolg    = 10^-8;   %tolerance lvl for GS convergence (should be < tolv)
maxiter = 1000;    %maximum iterations

NL_grid = 1;       %Non-linear grid? (Yes = 1, No = 0)


%% Steady state values
h_ss  = 0.33;
kh_ss = ((alpha*beta)/((1+gamma_z)-(1-delta)*beta))^(1/(1-alpha)); %K/h ratio
k_ss  = kh_ss*h_ss;
c_ss  = k_ss^alpha*h_ss^(1-alpha) - k_ss*((1+gamma_z)*(1+gamma_n)-(1-delta));
psi   = (1-alpha)*kh_ss^alpha*(1-h_ss)/c_ss;  %calibrate disutility of work
y_ss  = k_ss^alpha*h_ss^(1-alpha);
x_ss  = k_ss*((1+gamma_z)*(1+gamma_n)-(1-delta));
ss_check=y_ss-x_ss-c_ss;

%% Discretizing the capital grid
% Note: the K grid will be crucial to the labor choice. There could be k
% for which the labor choice is not defined. It must be kept close to the SS
kmin    = 0.5*k_ss;
kmax    = 2*k_ss;

if NL_grid == 1  %non-linear grid
    K = linspace(0,log(kmax-kmin+1),nk);
    K = exp(K)-1+kmin;
else             %linear grid
    K = linspace(kmin,kmax,nk);
end


%% Discretizing the z grid
[Z,zprob] = tauchen(ns,mu,rho,std_z,1);          % approximating the exogenous tech shock by a Markov chain
Z = exp(Z);                                       % to get a mean 1 process

%% Value Function iteration
% Initial values and pre-allocation
h = zeros(1,nk);                        % optimal h given k and c (golden search)
hh = zeros(ns,nk);                      % pre-allocation for the labor policy function
gg = zeros(ns,nk);                      % pre-allocation capital choice function
u = log(c_ss)+psi.*log(1-h_ss);         % utility at the steady state

V0 = ones(ns,nk)*u;                     % initial guess for the value function (the SS). This is necessary so that K includes the SS.
V1 = zeros(ns,nk);                      % pre-allocation for next iteration

% Initialize the VF loop
iterv  = 1;          %initialize iteration counter VF
diffv = tolv + 1;   %criterion for convergence

% Golden rule search with interpolation
tStart_v = tic;
while (diffv > tolv && iterv < maxiter)
    for is = 1:1:ns
        for ik = 1:1:nk                  
        
        % Given a value for k and z today, what is the value function? Use Bellman equation
        % Golden-section search over savings
        
        %extreme values of capital
        k1 = eps; % consume all the capital
        k4 = (K(ik)^alpha*(Z(is)*1).^(1-alpha) + (1-delta)*K(ik))/((1+gamma_n)*(1+gamma_z)); % save all available resources
        
        %initialize golden search loop
        diffg = tolg + 1; % golden search initialization
        iterg = 1;
        
        while (diffg > tolg && iterg < maxiter)
            %apply the golden rule to close
            klow = k1 + ((3.0-sqrt(5.0))/2.0)*(k4-k1);
            khigh = k1 + ((sqrt(5.0)-1.0)/2.0)*(k4-k1);

            %guarantee non-negativity of capital
            klow = max(klow,1e-8);
            khigh = max(khigh,1e-8);
            
            %find the implied labor supply given the choice of savings
            hlow = froot(K(ik),klow,Z(is),alpha,gamma_n,gamma_z,delta,psi); % implied labor for h2, given savings
            hhigh = froot(K(ik),khigh,Z(is),alpha,gamma_n,gamma_z,delta,psi); % implied labor for h3, given savings
            
            %find the implied consumption given the choice of savings and hours
            clow = K(ik)^alpha*(Z(is)*hlow).^(1-alpha) + (1-delta)*K(ik) - klow*(1+gamma_n)*(1+gamma_z);
            chigh = K(ik)^alpha*(Z(is)*hhigh).^(1-alpha) + (1-delta)*K(ik) - khigh*(1+gamma_n)*(1+gamma_z);
            
            %find the VF value given c and h using interpolation
            vlow = log(clow) + psi*log(1-hlow) + (1+gamma_n)*beta.*(zprob(is,:)*interp1(K,V0',klow,'spline')'); % value function given c2 and h2
            vhigh = log(chigh) + psi*log(1-hhigh) + (1+gamma_n)*beta.*(zprob(is,:)*interp1(K,V0',khigh,'spline')'); % value function given c3 and h3
        
            %update bounds
            if (vlow < vhigh)
                k1=klow; 
            else
                k4=khigh;
            end

            %update GS iteration
            iterg = iterg + 1;
            diffg=norm(k4-k1);
        end
        V1(is,ik) = (vhigh + vlow)/2;           % value function
        gg(is,ik) = (khigh + klow)/2;           % capital choice function
        hh(is,ik) = (hhigh + hlow)/2;           % labor choice function
                                
        end
    end

    %check VF convergence
    iterv = iterv + 1;
    diffv = norm(V1 - V0);  % check whether a fixed point has been reached
    if rem(iterv,10) == 0  %show convergence pattern every 10 iterations
        disp(['VF iter = ', num2str(iterv), '. Norm= ',num2str(diffv)])
    end

    %update new VF
    V0 = V1;
end
tEnd_v = toc(tStart_v);

%Print infos about VFI
if iterv < maxiter
    disp(['Value function converged in ', num2str(iterv), ' iterations :) Time taken: ',num2str(tEnd_v)])
else
    disp(['Value function did not converge :( Time taken: ',num2str(tEnd_v) ])
end


%% Plot Value Function and Policy Function
figure(1)  %Value Function
plot(K,V1);
title('Value Function')
xlabel('k');
ylabel('V');
legend('z_{low}','z_{high}','Location','southeast');

figure(2)  %savings policy function
plot(K,gg);
xlabel('k');
ylabel('k')
hline = refline(1,0);
hline.Color = 'black';
hline.LineStyle = '--';
title('Capital policy function');
legend('z_{low}','z_{high}','Location','southeast')


figure(3)  %labor supply policy function
plot(K,hh);
xlabel('k');
ylabel('h');
title('Labor supply policy function')
legend('z_{low}','z_{high}','Location','southwest')


%% Simulation
%Set-up
rng(75);      %random number seed
simul = 200;  %#periods to simulate

%Simulate z
Zsim     = ones(simul+1,1)*NaN;  %storage values of [i,z(i)]
[~, col] = size(Z);
Zsim(1,1) = col; %first state
for i = 2:simul+1
  this_step_dist = zprob(Zsim(i-1,1),:); %Prob at state
  cumulative_distribution = cumsum(this_step_dist);
  random = rand(); %generate a random number in [0,1]
  Zsim(i,1) = find(cumulative_distribution>random,1); %get the first to be higher in the cdf
end
Zsim = Z(Zsim); %convert state number to value of z

Ksim = zeros(1,simul+1);  % capital (includes simulations and steady state capital)
Ksim(1) = k_ss*.95;       % start from the steady state
Hsim = zeros(1,simul);    % labor supply
for i = 1:1:simul 
    Ksim(i+1) = spline(K,gg(Z==Zsim(i),:)',Ksim(i));      % simulating capital choices given the shock in each period (K_t+1)
    Hsim(i) = spline(K,hh(Z==Zsim(i),:)',Ksim(i));
end
Csim = Ksim(1:end-1).^alpha .* (Zsim(1:end-1)'.*Hsim).^(1-alpha) + (1-delta)*Ksim(1:end-1) - Ksim(2:end)*(1+gamma_n)*(1+gamma_z); % using the resource constraint to find implied consumption 
Ysim = Ksim(1:end-1).^alpha .* (Zsim(1:end-1)'.*Hsim).^(1-alpha);                                         % using the production function to find implied output

%% Simulation Plots
figure(4)
subplot(321)
plot(Zsim(1:end-1),'LineWidth',1,'Color','black')
title('Income shock')
xlabel('time')
ylabel('z')
subplot(322)
plot(Csim,'LineWidth',1,'Color','black')
title('Consumption')
xlabel('time')
ylabel('c')
subplot(323)
plot(Ysim,'LineWidth',1,'Color','black')
title('Output')
xlabel('time')
ylabel('y')
subplot(324)
plot(Ksim(1:end-1),'LineWidth',1,'Color','black')
title('Capital')
xlabel('time')
ylabel('k')
subplot(325)
plot(Hsim(1:end-1),'LineWidth',1,'Color','black')
title('Labor Supply')
xlabel('time')
ylabel('h')
