%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The Huggett model (using EGM)
% Tiago Bernardino, IIES - Stockholm University
% November, 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Model Parameters
eis  = 1.2;        %elasticity of intertemporal substitution
r    = 0.055/4;     %interest rate
beta = 0.985;       %discount factor
mu   = 1;          %constant of the AR(1) process
rho  = 0.95;       %persistency of the state process
std  = 0.1;        %std dev of the income shock


%% Computational parameters
amin = 0;
amax = 200;
na   = 500;
ns   = 3;
maxiter = 10000;
maxiter_dist = 50000;
tolegm  = 10^-8;
toldist = 10^-10;


%% Discretize assets and income
% Asset grid (non-linear)
ubar = log(1+log(1+amax-amin));

Agrid = linspace(0,ubar,na);
Agrid = amin + exp(exp(Agrid)-1)-1;

% Income grid
w = 0.5 + rho/4;                %auxiliar
sigmaZ = std/sqrt(1-rho^2);     %auxiliar
baseSig = w*std +(1-w)*sigmaZ;

[Y,yprob] = tauchenhussey(ns,mu,rho,std,baseSig);

clear ubar w sigmaZ baseSig


%% Initial guess
coh = Agrid * (1+r) + Y;
c   = 0.05 * coh;
Va = (1+r) * c.^(-1/eis);

%% Bacward iteration

% Initialize loop
iter = 0;
diffegm = tolegm + 1;

tStart = tic;
while diffegm > tolegm && iter < maxiter
iter = iter + 1;

% Step 1: Discounting and expectations
Wa = (beta * yprob) *Va;

% Step 2: Solving for asset policy using FOC
c_endog = Wa.^(-eis);
coh = (1+r)*Agrid + Y;

a = zeros(ns,na);
for is = 1:ns
    a(is,:) = interp1(c_endog(is,:) + Agrid,Agrid,coh(is,:),"linear",'extrap');
end

% Step 3: Enforcing the borrowing constraint and get consumption
a = max(a, amin);
c = coh - a;

% Step 4: Using the EC to recover the derivative of the VF
Va = (1+r) * c.^(-1/eis);

if iter > 3
    diffegm = max(abs(a_old - a),[],'all');
end
a_old = a;

end
tEnd = toc(tStart);

% Some controls along the Market clearing convergence regarding VFI
if iter < maxiter
    disp(['EGM converged in ', num2str(iter), ' iterations :) Time taken: ',num2str(tEnd)])
else
    disp(['EGM did not converge :( Time taken: ',num2str(tEnd) ])
end

% Store policy function
gg = a;
cc = c;

% Note that one can obtain the VF by runing a VFI algorithm using the
% policy function solution from EGM as guess.


%% Compute the distribution

% I. Get lotery for the policy function

% Step 1: find the index such that gg lies between Agrid_i and Agrid_(i+1)
a_i = zeros(ns,na);
for is = 1:ns  %find the closest index
    a_i(is,:) = knnsearch(Agrid',gg(is,:)');
end

test = Agrid(a_i) > gg;   %replace such that a_i is the lower bound
a_i(test) = a_i(test)-1;  

a_i  = a_i.*(a_i<na) + (na-1)*(a_i>=na);   %replace such that a_i is not at amax

% Step 2: obtain lottery probabilities pi
a_pi = (Agrid(min(a_i+1,na)) - gg)./(Agrid(min(a_i+1,na)) - Agrid(a_i));

test = Agrid(min(a_i+1,na)) == Agrid(a_i);  %make sure the denominator is not 1
a_pi(test) = 1;

test = gg > Agrid(min(a_i+1,na));   %for savings > amax it assigns to the highest grid point
a_pi(test) = 1;

a_pi(a_pi>1) = 1;   % ensure probability smaller than 1

% II. Initial guess for the distribution: we set the initial guess to be the
% right one for s and uniform for a
ystat = yprob^10000;
D = repmat(ystat(1,:)', [1, na]);
D = D./sum(D,'all');

% III. Initialize loop
iter = 0;
diffdist = toldist + 1;

% IV. Iterate until convergence 
tStart_dist = tic;
while diffdist > toldist && iter < maxiter_dist
iter = iter + 1;

Dnew = zeros(ns,na);
for is=1:ns
    for ia=1:na
        %send pi(s,a) of the mass to gridpoint i(s,a)
        Dnew(is, a_i(is,ia)) = Dnew(is, a_i(is,ia)) + a_pi(is,ia)*D(is,ia);
            
        %send 1-pi(s,a) of the mass to gridpoint i(s,a)+1
        Dnew(is, a_i(is,ia)+1) = Dnew(is, a_i(is,ia)+1) + (1-a_pi(is,ia))*D(is,ia);
    end
end
Dnew = yprob*Dnew;

% Check convergence
diffdist = max(abs(Dnew - D),[],'all');

D = Dnew;
D = D./sum(D,'all');
end
DD = D;

tEnd_dist = toc(tStart_dist);

if iter < maxiter_dist
    disp(['Stationary distribution found in ', num2str(iter), ' iterations :) Time taken: ',num2str(tEnd_dist)])
else
    disp(['Stationary distribution not found :( Time taken: ',num2str(tEnd_dist)])
end

%% Plot Policy Functions

%savings policy function
figure(1)  
plot(Agrid,gg);
xlabel('a');
ylabel('a_{t+1}')
hline = refline(1,0);
hline.Color = 'black';
hline.LineStyle = '--';
title('Savings policy function');
legend('y_{low}','y_{high}','Location','southeast')

%consumption policy function
figure(2)  
plot(Agrid,cc);
xlabel('a');
ylabel('c_{t}')
title('Consumption policy function');
legend('y_{low}','y_{high}','Location','southeast')

%SS distribution
figure(3)
plot(Agrid,sum(DD,1));
xlabel('a');
ylabel('Probability')
title('Agents distribution in the SS');