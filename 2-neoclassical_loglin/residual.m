function R = residual(Z,param)
%RESIDUAL FUNCTION: residual for the stochastic growth model.
% Inputs:
%       Z: vector (log_k_{t+2},log_k_{t+1},log_k_t,log_z_{t+1},log_z_t)
%       param: vector (alpha,beta,delta,sigma)


%% Parameters
alpha   = param(1);  %capital share
beta    = param(2);  %discount factor
delta   = param(3);  %depreciation rate
sigma   = param(4);  %coefficient of risk aversion

%% Variables
k2         = exp(Z(1));
k1         = exp(Z(2));
k          = exp(Z(3));

z1         = exp(Z(4));
z          = exp(Z(5));

y          = z*k^alpha;
y1         = z1*k1^alpha;

c          = y+(1-delta)*k-k1;
c1         = y1+(1-delta)*k1-k2;

%% Outcome
R          = beta*(c1/c)^(-sigma)*(alpha*z1*k1^(alpha-1)+(1-delta))-1;