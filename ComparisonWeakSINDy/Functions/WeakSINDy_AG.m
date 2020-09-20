%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WSINDy
%%%%%%%%%%%% 
%%%%%%%%%%%% Copyright 2020, All Rights Reserved
%%%%%%%%%%%% Code by Daniel A. Messenger
%%%%%%%%%%%% For Paper, "Weak SINDy: Galerkin-based Data-Driven Model
%%%%%%%%%%%% Selection"
%%%%%%%%%%%% by D. A. Messenger and D. M. Bortz


function w_sparse = WeakSINDy_AG(tobs,xobs,Theta,n,lambda,gamma,r_whm,wsindy_params)
w_sparse = zeros(size(Theta,2),n);
mats = cell(n,1);
ps_all = [];
ts_grids = cell(n,1);
RTs = cell(n,1);
Ys = cell(n,1);
Gs = cell(n,1);
bs = cell(n,1);

for i=1:n
    [Y,grid_i] = adaptive_grid(tobs,xobs(:,i),wsindy_params);
    [V,Vp,ab_grid,ps] = VVp_build_adaptive_whm(tobs,grid_i, r_whm, {0,inf,0});  %{pow,nrm,ord}. ord=0, ||phi||, ord=1, ||phi'||
    ps_all = [ps_all;ps];
    mats{i} = {V,Vp};
    ts_grids{i} = ab_grid;
    Ys{i} = Y;
    Cov = Vp*Vp'+10^-12*eye(size(V,1));
    [RT,~] = chol(Cov);
    RT = RT';
    G = RT \ (V*Theta);
    b = RT \ (Vp*xobs(:,i));
    RTs{i} = RT;
    Gs{i} = G;
    bs{i} = b;
    w_sparse(:,i) = sparsifyDynamics(G,b,lambda,1,gamma);
end
