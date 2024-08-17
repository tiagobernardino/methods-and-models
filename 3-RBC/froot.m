function h = froot(k,kp,z,theta,gamma_n,gamma_z,delta,psi)
    % Labor supply equation
    function out = ls(h)
        out=z^(1-theta)*(k/h)^theta*(1-theta)-((k^theta*(z*h)^(1-theta)+(1-delta)*k-(1+gamma_z)*(1+gamma_n)*kp)*psi)/(1-h);
    end

    fun=@ls;

    h=fzero(fun,[0.000001 .999999]);
end