%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WSINDy: function for returning basis functions of the form
%%%%%%%%%%%% found in the paper below.
%%%%%%%%%%%% 
%%%%%%%%%%%% Copyright 2020, All Rights Reserved
%%%%%%%%%%%% Code by Daniel A. Messenger
%%%%%%%%%%%% For Paper, "Weak SINDy: Galerkin-based Data-Driven Model
%%%%%%%%%%%% Selection"
%%%%%%%%%%%% by D. A. Messenger and D. M. Bortz

function [g,gp] = basis_fcn(p,q)

    g = @(t,t1,tk) ... 
        (t-t1).^(max(p,0)).*(tk-t).^(max(q,0)).*(q>0).*(p>0) + ...
        (1-2*abs(t-(t1+tk)/2)./(tk-t1)).*(q==0).*(p==0) +... 
        sin(p*pi./(tk-t1).*(t-t1)).*(q<0).*(p>0) +...
        (p==-1).*(q==-1);
    gp = @(t,t1,tk) ...
        (t-t1).^(max(p-1,0)).*(tk-t).^(max(q-1,0)).*((-p-q)*t+p*tk+q*t1).*(q>0).*(p>0) +...
        -2*sign(t-(t1+tk)/2)./(tk-t1).*(q==0).*(p==0) +...
        p*pi./(tk-t1).*cos(p*pi./(tk-t1).*(t-t1)).*(q<0).*(p>0)+ ...
        0*(p==-1).*(q==-1);
    if and(p>0,q>0)
        gp = @(t,t1,tk) gp(t,t1,tk)./abs(g((q*t1+p*tk)/(p+q),t1,tk));
         g = @(t,t1,tk) g(t,t1,tk)./abs(g((q*t1+p*tk)/(p+q),t1,tk));
    end
        
end
