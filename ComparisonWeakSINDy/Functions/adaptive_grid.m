%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WSINDy: function for building test function centers adaptively 
%%%%%%%%%%%% 
%%%%%%%%%%%% Copyright 2020, All Rights Reserved
%%%%%%%%%%%% Code by Daniel A. Messenger
%%%%%%%%%%%% For Paper, "Weak SINDy: Galerkin-based Data-Driven Model
%%%%%%%%%%%% Selection"
%%%%%%%%%%%% by D. A. Messenger and D. M. Bortz

function [Y,final_grid] = adaptive_grid(t,xobs,params)

if ~exist('params','var')
    index_gap = 16;
    K = max(floor(length(t)/50),4);
    p = 2;
    tau = 1;
else
    index_gap = params{1};
    K = params{2};
    p = params{3};
    tau = params{4};
end

M = length(t);

[g,gp] = basis_fcn(p,p);
[~,Vp_row] = tf_mat_row(g,gp,t,1,1+index_gap,{1,1,0});
Vp_diags = repmat(Vp_row(1:index_gap+1),M-index_gap,1);
Vp =  spdiags(Vp_diags, 0:index_gap,M-index_gap,M);
weak_der = Vp*xobs; % "weak convolution" to get approximate derivative
weak_der = [zeros(floor(index_gap/2),1);weak_der;zeros(floor(index_gap/2),1)];

Y = abs(weak_der);
Y = cumsum(Y);
Y = Y/Y(end);
Y = tau*Y+ (1-tau)*linspace(Y(1),Y(end),length(Y))';

U =linspace(Y(floor(index_gap/2)),Y(end-ceil(index_gap/2)+1),K+2);
final_grid = zeros(1,K);

for i=1:K
    final_grid(i) = find(Y-U(i+1)>=0,1);
end

final_grid = unique(final_grid);
end

function [V_row,Vp_row] = tf_mat_row(g,gp,t,t1,tk,param)
    N = length(t);
    
    if ~exist('param','var')
        gap=1;
        nrm=inf;
        ord=0; 
    else
        gap = param{1};
        nrm = param{2};
        ord = param{3};
    end
    
    [a,b] = size(t);
    if a>b
        t = reshape(t,b,a);
    end
    
    if t1>tk
        tk_temp = tk;
        tk = t1;
        t1 = tk_temp;
    end
    
    V_row = zeros(1, N);
    Vp_row = V_row;

    t_grid = t(t1:gap:tk);
    dts = diff(t_grid);
    w = 1/2*([dts 0]+[0 dts]);
    
    V_row(t1:gap:tk) = g(t_grid,t(t1),t(tk)).*w;
    Vp_row(t1:gap:tk) = -gp(t_grid,t(t1),t(tk)).*w;
    Vp_row(t1) = Vp_row(t1) - g(t(t1),t(t1),t(tk));
    Vp_row(tk) = Vp_row(tk) + g(t(tk),t(t1),t(tk));
    
    if ord == 0
        scale_fac = norm(V_row(t1:gap:tk),nrm);
    elseif ord ==1
        scale_fac = norm(Vp_row(t1:gap:tk),nrm);
    else
        scale_fac = mean(dts);
    end
    
    Vp_row = Vp_row/scale_fac;
    V_row = V_row/scale_fac;

end
