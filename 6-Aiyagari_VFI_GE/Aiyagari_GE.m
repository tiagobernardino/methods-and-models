%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The Aiyagari model (using VFI)
% Tiago Bernardino, IIES
% January, 2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close all; clc

%% Model Parameters (build in structure)
alpha      = 0.36;      %capital share
delta      = 0.06;      %depreciation rate
sigma      = 2;         %coefficient of risk aversion
mu         = 0;         %constant of the AR(1) process
rho_y      = 0.95;      %persistency of the state process
std_s      = 0.015^0.5; %std dev of the income shock
Kbar       = 0;         %zero net-supply
rho_A      = 0.9;       %persistency of TFP
std_A      = 0.012;     %std dev of the TFP shock process
K_Y_ratio  = 5;         %capital to output ratio


%% Computational parameters
ns         = 3;         %#possible income states in the economy
nk         = 500;       %#k grid points
kmin       = eps;       %borrwoing constraint
kmax       = 24;        %maximum assets
tolk       = 10^-4;     %tolerance level GE loop
tolv       = 10^-10;    %tolerance level VF loop
toldist    = 10.^-8;    %tolerance stationary distribution
maxiter    = 5000;      %maximum #iterations for VFI
maxiter_MC = 100;       %maximum #iterations for market clearing
vmin       = -1e10;     %set a very small number
NL_grid    = 1;         %Produce non-linear grid?
nk_fine    = 1000;
curv_fine  = 4;


%% discretize AR(1) productivity process
w = 0.5 + rho_y/4;               %auxiliar
sigmaZ = std_s/sqrt(1-rho_y^2);  %auxiliar
baseSig = w*std_s +(1-w)*sigmaZ; %auxiliar
[S,S_prob] = tauchenhussey(ns,mu,rho_y,std_s,baseSig);
y          = exp(S);             %income process
y_prob     = S_prob;             %income transition matrix

clear w sigmaZ baseSig


%% Capital grids
if NL_grid == 1  %non-linear grid
    K = linspace(0,log(kmax-kmin+1),nk);
    K = exp(K)-1+kmin;
else  %linear grid
    K = linspace(kmin,kmax,nk);
end

K_fine = kmin+(kmax-kmin)*(linspace(0,1,nk_fine).^curv_fine);


%% GE and VFI
beta_min = 0.98;  %lower bound
beta_max = 0.99;  %upper bound

beta = (beta_min+beta_max)/2;         %initial guess;
beta_store = ones(maxiter_MC,1)*NaN;  %store the beta iterations

r = alpha * K_Y_ratio^(-1) - delta;           %implied r given K/Y
K_L_ratio = (alpha/(r+delta))^(1/(1-alpha));  %capital to labor ratio
w = (1-alpha) * K_L_ratio^(alpha);            %implied wage rate

rIter_start = tic;
for i0 = 1:maxiter_MC
beta_store(i0) = beta;  %store the discount factor

% Build consumption and utility grid
C = repmat(K',ns,nk)*(1+r) + kron(y,ones(nk))*w - repmat(K,ns*nk,1);
U = (C.^(1-sigma)-1) ./ (1-sigma);
U(C<=0) = vmin;

% Initial guess of VF
if i0==1
    V = zeros(ns*nk,1);
    for j=1:ns
    % Set such that a=a'
    vtemp = U((j-1)*nk+1:j*nk,:);
    V((ns-1)*nk+1:ns*nk)=diag(vtemp)/(1-beta);
    end
end
TV = V;

% Value Function iteration
tStart_v = tic;
for i1=1:maxiter
    % Take expectation of value function
    EV = y_prob * reshape(V,nk,ns)';
    % Find maximum and policy
    [TV, pol] = max( U + beta * kron(EV,ones(nk,1)), [], 2);

    % Check for convergence of the value function 
    dV = max(abs(TV-V));
    if dV < tolv
        break
    else
        V=TV;
    end

     if i0==1
     % Print some output first time around
     disp(['VF iter = ', num2str(i1), '. Norm= ',num2str(dV)])
     end
end
tEnd_v = toc(tStart_v);

% Some controls along the Market clearing convergence regarding VFI
if i1 < maxiter
    disp(['Value function converged in ', num2str(i1), ' iterations :) Time taken: ',num2str(tEnd_v)])
else
    disp(['Value function did not converge :( Time taken: ',num2str(tEnd_v) ])
end

% Compute the stationary distribution
tStart_dist = tic;
PP = zeros(ns*nk);  %Transition matrix (a,s) -> (a',s')
a_eye = eye(nk);
for is=1:ns
    for isp=1:ns
    index1 = (is-1)*nk+1:is*nk;
    index2 = (isp-1)*nk+1:isp*nk;
    PP(index1,index2) = y_prob(is,isp)*a_eye(pol(index1),:);
    end
end

%stationary distribution - eigenvectors method
[eigV,eigD] = eig(PP');               %Get the eigenvectors and eigenvalues
i_eig1      = dsearchn(diag(eigD),1); %Get the first unit eigenvalue
lambda      = eigV(:,i_eig1);         %Get the correspondent eigenvector
lambda      = lambda/sum(lambda);     %Normalize the distribution

tEnd_dist = toc(tStart_dist);
disp(['Distribution found :) Time taken: ',num2str(tEnd_dist)])

%Asset market clearing
kpol = K(pol);       %Find actual asset choice
Capital = kpol * lambda;   %Get the implied asset distribution
K_Y_ratio_implied = Capital^(1-alpha);

diff_MC = K_Y_ratio_implied-K_Y_ratio;     %Check for asset market converge

disp(['Market clearing iter = ', num2str(i0), '. K/Y implied= ',num2str(K_Y_ratio_implied),' . discount factor: ', num2str(beta)])

if abs(diff_MC) < tolk
     break
end 
 
if diff_MC>0
    beta_max = beta;
else
    beta_min = beta;
end

beta = (beta_max+beta_min)/2; % bisection new bounds
end
rIter_end = toc(rIter_start);

disp(' ')
if i0<maxiter_MC
    disp(['Market clearing converged in ', num2str(i0), ' iterations'])
    disp(['Discount factor: ',num2str(beta)])
else
    disp('Market clearing did not converge')
end

disp(['TOTAL TIME taken for program to run: ',num2str(rIter_end), ' seconds' ])

%% Reshape objects
income_RHS       = ones(ns,nk)*NaN;       %income - RHS
gg               = ones(ns,nk)*NaN;       %asset policy function
gg_fine          = ones(ns,nk_fine)*NaN;  %asset policy function (finer grid)
VV               = ones(ns,nk)*NaN;       %value function
lambda_resh      = ones(ns,nk)*NaN;       %distribution

% Reshape into (ns x nk) matrices
for is = 1:ns 
    for ik = 1:nk
        VV(is,ik)          = V(ik+(is-1)*nk);
        income_RHS(is,ik)  = w*y(is) + (1+r)*K(ik);
        gg(is,ik)          = kpol(ik+(is-1)*nk);
        lambda_resh(is,ik) = lambda(ik+(is-1)*nk);
    end
end
cc = income_RHS - gg;
lambda = lambda_resh;