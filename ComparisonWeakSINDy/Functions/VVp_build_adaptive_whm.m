%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% WSINDy: function for building integration matrices to integrate 
%%%%%%%%%%%% against test functions and derivatives 
%%%%%%%%%%%% 
%%%%%%%%%%%% Copyright 2020, All Rights Reserved
%%%%%%%%%%%% Code by Daniel A. Messenger
%%%%%%%%%%%% For Paper, "Weak SINDy: Galerkin-based Data-Driven Model
%%%%%%%%%%%% Selection"
%%%%%%%%%%%% by D. A. Messenger and D. M. Bortz

function [V,Vp,ab_grid,ps] = VVp_build_adaptive_whm(t,centers,r_whm,param)
    if ~exist('param','var')
        param = {1,2,1}; 
    end
    
    [a,b] = size(t);   
    if a>b
        t = reshape(t,b,a);
    end
    N = length(t);
    M = length(centers);
    V = zeros(M,N);
    Vp = V;
    ab_grid = zeros(M,2);
    ps = zeros(M,1);
    
    [p,a,b] = test_fcn_param(r_whm,t(centers(1)),t);
    if b-a < 10    %%%%%%%% if support is less than 10 points, enforce that support is exact 10 points
        center = (a+b)/2;
        a = max(1,floor(center-5));
        b = min(ceil(center+5),length(t));
    end
    [g,gp] = basis_fcn(p,p);
    [V_row,Vp_row] = tf_mat_row(g,gp,t,a,b,param);
    V(1,:) = V_row;
    Vp(1,:) = Vp_row;
    ab_grid(1,:) = [a b];
    ps(1) = p;

    for k=2:M
        cent_shift = centers(k)-centers(k-1);
        b_temp = min(b + cent_shift,length(t));
        if and(a>1,b_temp<length(t))
            a = a + cent_shift;
            b = b_temp;
            V_row = circshift(V_row,cent_shift);
            Vp_row = circshift(Vp_row,cent_shift);
        else
            [p,a,b] = test_fcn_param(r_whm,t(centers(k)),t);
            if b-a < 10
                center = (a+b)/2;
                b = min(ceil(center+5),length(t));
                a = max(1,floor(center-5));
            end
            [g,gp] = basis_fcn(p,p);
            [V_row,Vp_row] = tf_mat_row(g,gp,t,a,b,param);
        end            
        V(k,:) = V_row;
        Vp(k,:) = Vp_row;
        ab_grid(k,:) = [a b];
        ps(k) = p;
    end
end

function [V_row,Vp_row] = tf_mat_row(g,gp,t,t1,tk,param)
    N = length(t);
    
    if ~exist('param','var')
        pow=1;
        nrm=inf;
        ord=0;
        gap = 1;
    else
        pow = param{1};
        nrm = param{2};
        ord = param{3};
        gap =1;
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
    
    if pow~=0
        if ord == 0
            scale_fac = norm(V_row(t1:gap:tk),nrm)^pow;
        elseif ord ==1
            scale_fac = norm(Vp_row(t1:gap:tk),nrm)^pow;
        else
            scale_fac = mean(dts)^pow;
        end
        Vp_row = Vp_row/scale_fac;
        V_row = V_row/scale_fac;    
    end
end