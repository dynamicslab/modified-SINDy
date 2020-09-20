%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WSINDy
%%%%%%%%%%%% 
%%%%%%%%%%%% Copyright 2020, All Rights Reserved
%%%%%%%%%%%% Code by Daniel A. Messenger
%%%%%%%%%%%% For Paper, "Weak SINDy: Galerkin-based Data-Driven Model
%%%%%%%%%%%% Selection"
%%%%%%%%%%%% by D. A. Messenger and D. M. Bortz


function [p,a,b] = test_fcn_param(r,c,t,tol)

if ~exist('tol','var')
    tol = 16;
end

dt = t(2)-t(1);
r_whm = r*dt;
A = log2(10)*tol;

gg = @(s) -s.^2.*((1-(r_whm./s).^2).^A-1);
hh = @(s) (s-dt).^2;
ff = @(s) hh(s)-gg(s);

s = fzero(ff,[r_whm, r_whm*sqrt(A)+dt]);

p = min(ceil(max(-1/log2(1-(r_whm/s)^2),1)),200);
a = find(t >= (c-s),1);
if c+s > t(end)
    b = length(t);
else
    b = find(t >= (c+s),1);
end
end

% r = 3;
% dt = 0.01;
% r_whm = r*dt;
% tol=16;
% A = log2(10)*tol;
% 
% gg = @(s) -s.^2.*((1-(r_whm./s).^2).^A-1);
% hh = @(s) (s-dt).^2;
% ff = @(s) hh(s)-gg(s);
% 
% s1 = fzero(ff,[r_whm, r_whm*sqrt(A)+dt])
% ss = linspace(r_whm,r_whm*sqrt(A)+dt,100);
% 
% plot(ss,gg(ss), ss,hh(ss),ss,ff(ss))
% legend({'gg','hh','ff'})
