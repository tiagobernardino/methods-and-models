%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The Huggett model in GE (using VFI with grid search and bisection on r)
% Tiago Bernardino, IIES - Stockholm University
% August, 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc

%% Model Parameters
beta    = 0.96;  %discount factor
sigma   = 1.2;         %coefficient of risk aversion
mu      = 0;         %constant of the AR(1) process
rho     = 0.95;      %persistency of the state process
std_s   = 0.025^0.5; %std dev of the income shock
Abar    = 10;         %zero net-supply


%% Computational parameters
ns      = 2;         %#possible states in the economy
amin    = 0;         %borrwoing constraint
amax    = 24;        %maximum assets
na      = 1000;      %#a grid points
NL_grid = 1;         %Produce non-linear grid?
tola       = 10^-2;  %tolerance level GE loop
tolv       = 10^-5;  %tolerance level VF loop
told       = 10^-8;  %tolerance level distribution loop
maxiter_VF = 5000;   %maximum #iterations for VFI
maxiter_MC = 100;    %maximum #iterations for market clearing
vmin       = -1e10;  %set a very small number

%% Income process
w = 0.5 + rho/4;                    %auxiliar
sigmaZ = std_s/sqrt(1-rho^2);       %auxiliar
baseSig = w*std_s +(1-w)*sigmaZ;    %input of tauchenhussey.m
[y,y_prob] = tauchenhussey(ns,mu,rho,std_s,baseSig);

clear w sigmaZ baseSig

%% Asset grid
if NL_grid == 1
    a = linspace(0,log(amax-amin+1),na);
    a = exp(a)-1+amin;
else
    a = linspace(amin,amax,na);
end

%% GE + VFI loops
disp('GE loop initialized');

rmin = -0.05;  %lower bound
rmax = 0.05;  %upper bound
r = (rmin+rmax)/2;  %initial guess
r_store = ones(maxiter_MC,1)*NaN;  %store the interest rate iterations
a_eye = eye(na);

rIter_start = tic;
for i0 = 1:maxiter_MC
r_store(i0) = r;  %store the interest rate
    
% Build consumption and utility grid
C = repmat(a',ns,na)*(1+r) + kron(y,ones(na)) - repmat(a,ns*na,1);
U = (C.^(1-sigma)-1) ./ (1-sigma);
U(C<=0) = vmin;

% Initial guess of VF
if i0==1
    V = zeros(ns*na,1);
    for j=1:ns
    % Set such that a=a'
    vtemp = U((j-1)*na+1:j*na,:);
    V((ns-1)*na+1:ns*na)=diag(vtemp)/(1-beta);
    end
end
TV = V;

% Value Function iteration
tStart_v = tic;
for i1=1:maxiter_VF
    % Take expectation of value function
    EV = y_prob * reshape(V,na,ns)';
    % Find maximum and policy for wach state
    [TV, pol] = max( U + beta * kron(EV,ones(na,1)), [], 2);

    % Check for convergence of the value function 
    dV = max(abs(TV-V));
    if dV < tolv
        break
    else
        V=TV;
    end

    if i0==1
    % Print some output first time around VFI
    disp(['VF iter = ', num2str(i1), '. Norm= ',num2str(dV)])
    end
end
tEnd_v = toc(tStart_v);

% Some controls along the Market clearing convergence regarding VFI
if i1 < maxiter_VF
    disp(['Value function converged in ', num2str(i1), ' iterations :) Time taken: ',num2str(tEnd_v)])
else
    disp(['Value function did not converge :( Time taken: ',num2str(tEnd_v) ])
end


% Compute the stationary distribution
tStart_dist = tic;
PP = zeros(ns*na);  %Transition matrix (a,s) -> (a',s')
for is=1:ns
    for isp=1:ns
    index1 = (is-1)*na+1:is*na;
    index2 = (isp-1)*na+1:isp*na;
    PP(index1,index2) = y_prob(is,isp)*a_eye(pol(index1),:);
    end
end

% %stationary distribution - brute force method
% lambda = PP^2000;
% lambda = transpose(lambda(1,:));

%stationary distribution - eigenvectors method
[eigV,eigD] = eig(PP');               %Get the eigenvectors and eigenvalues
i_eig1      = dsearchn(diag(eigD),1); %Get the first unit eigenvalue
lambda      = eigV(:,i_eig1);         %Get the correspondent eigenvector
lambda      = lambda/sum(lambda);     %Normalize the distribution

tEnd_dist = toc(tStart_dist);
disp(['Distribution found :) Time taken: ',num2str(tEnd_dist)])

%Asset market clearing
apol = a(pol);      %Find actual asset choice
A = apol * lambda;  %Get the asset distribution

excess_supply = Abar-A;        %Check for asset market converge

disp(['Market clearing iter = ', num2str(i0), '. Net supply= ',num2str(Abar-A),' . Interest rate: ', num2str(r)])

if abs(excess_supply) < tola
    break
end 

if excess_supply>0
    rmin = r;
else
    rmax = r;
end

r = (rmax+rmin)/2; % bisect new bounds
end
rIter_end = toc(rIter_start);

disp(' ')
if i0<maxiter_MC
    disp(['Market clearing converged in ', num2str(i0), ' iterations'])
    disp(['Equilibrium interest rate: ',num2str(r)])
else
    disp('Market clearing did not converge')
end

disp(['TOTAL TIME taken for program to run: ',num2str(rIter_end), ' seconds' ])

%% Plot Value Function and Policy Function
figure(1)  %Value Function
plot(a,V(1:1000), a, V(1001:2000));
title('Value Function')
xlabel('a');
ylabel('V');
legend('z_{low}','z_{high}','Location','southeast');

figure(2)  %savings policy function
plot(a,apol(1:1000),a,apol(1001:2000));
xlabel('a');
ylabel('a_{t+1}')
hline = refline(1,0);
hline.Color = 'black';
hline.LineStyle = '--';
title('Savings policy function');
legend('z_{low}','z_{high}','Location','southeast')

