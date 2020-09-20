function Xi = sparsifyDynamics(Theta,dXdt,lambda,n,gamma,M)
% Copyright 2015, All Rights Reserved
% Code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz
%
% Modified by Daniel A. Messenger, 2020 
%
% compute Sparse regression: sequential least squares

if ~exist('M','var')
    M = ones(size(Theta,2),1);
end

if  gamma == 0
    Theta_reg = Theta;
    dXdt_reg = dXdt;
else
    nn = size(Theta,2);
    Theta_reg = [Theta;gamma*eye(nn)];
    dXdt_reg = [dXdt;zeros(nn,n)];
end

Xi = M.*(Theta_reg \ dXdt_reg);  % initial guess: Least-squares
% lambda is our sparsification knob.
for k=1:15
    smallinds = (abs(Xi)<lambda);   % find small coefficients
    while size(find(smallinds)) == size(Xi(:)) % make sure zero vector not returned
        lambda = lambda/2;
        smallinds = (abs(Xi)<lambda);   % find small coefficients
    end
    Xi(smallinds)=0;                % and threshold
    for ind = 1:n                   % n is state dimension
        biginds = ~smallinds(:,ind);
        % Regress dynamics onto remaining terms to find sparse Xi
        Xi(biginds,ind) = M(biginds).*(Theta_reg(:,biginds)\dXdt_reg(:,ind));
    end
end