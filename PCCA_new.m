% PCCA distance learning algorithm proposed in "PCCA: A New Approach for
% Distance Learning from sparse pairwise constraints" cvpr 2012
% By Fei Xiong, 
%    ECE Dept, 
%    Northeastern University 
%    2013-11-04
% INPUT
%   X: N-by-d data matrix. Each row is a sample vector.
%   ix_pair: the index for pairwise constraints.
%   y: the annotation for pairwise constraints. {+1, -1}
%   option: algorithm options
%       beta: the parameter in generalized logistic loss function
%       d: the dimensionality of the projected feature
%       eps: the tolerence value.
% OUTPUT
%   Method: the structure contains the learned projection matrix and
%       algorithm parameters setup.
%   l_old: the objective value
%   AKA: the regularizer value
% UPDATE LOG:
%   05/26/2014 speeding up the initialize part
function [Method]= PCCA_new(X, ix_pair, y, option,w,svmpair,C,Method,K,KJ)
ix_pair= double(ix_pair);
if length(size(X)) >2 % input feature are covariance matrix
    [Method, l_old, AKA]= PCCA_Cov(X, ix_pair, y, option);
    return;
end

display(['begin PCCA ' option.kernel]);
beta= option.beta;
d = option.d;
eps = option.epsilon;
eta = 0.01;
if option.lambda>0
    W= eye(size(X,1))*option.lambda;
end

%%
%initialization
% A =randn(d, size(X,1))/1e4;
A=Method.P;
temp = K(ix_pair(:,1),:) - K(ix_pair(:,2),:);
temp = A*temp';
D = sum(temp.^2, 1);
AKA =trace(A*K*A');

ttemp=K(:,svmpair(:,1))-K(:,svmpair(:,2));
%   ttemp_new=1-w'*A*ttemp;
ttemp_new=w'*A*ttemp;
ttemp_new=max(0,1-ttemp_new);
sv=ttemp_new>0;
% l_old = sum(logistic_loss(D,y,option)) + option.lambda* AKA; %
l_old = sum(logistic_loss(D,y,option)) + option.lambda* AKA+...
       (1/2*w'*w+1/2*C*sum(ttemp_new.^2,2));
M=repmat(w,1,size(ttemp,2));
DD=diag(double(sv));
deno=(-C*M*DD*ttemp'+C*w*w'*A*ttemp*DD*ttemp');
cnt =0;
ppp=0;
while 1
    ppp=ppp+1
    L = -y'.*(1 -D); % there is a sign mistake in equation (3) and equation for L_n^t
                     % see reference [52]
    L = 1./ (1+exp(-beta*L));
    L = double(y'.*L);
    if option.lambda ==0
        Y = reshape(L*KJ,[size(X,1) size(X,1)]);
    else
        Y = reshape(L*KJ,[size(X,1) size(X,1)]) + W;
    end
% %     % optimization eta does not work
%     temp =reshape(sum(KJ),[size(X,1) size(X,1)]);
%     K=double(K);
%     temp = temp*K*Y*A'*A;
%     eta = trace(temp)/(2*trace(temp*Y));

%     A_new = A - 2*A *eta *Y;%*K
    A_new = A - eta *(2*A *Y+deno);
    temp = K(ix_pair(:,1),:) - K(ix_pair(:,2),:);
    temp = A_new*temp';
    D = sum(temp.^2, 1);
    AKA_new =trace(A_new*K*A_new');


    ttemp=K(:,svmpair(:,1))-K(:,svmpair(:,2));
    ttemp_new=w'*A_new*ttemp;
    ttemp_new=max(0,1-ttemp_new);
    sv=ttemp_new>0;
%     l_new = sum(logistic_loss(D,y,option))+ option.lambda * AKA_new; % + option.lambda* trace(A*K*A')
    l_new = sum(logistic_loss(D,y,option))+ option.lambda * AKA_new+...
            (1/2*w'*w+1/2*C*sum(ttemp_new.^2,2));
    l_new

    M=repmat(w,1,size(ttemp',1));
    DD=diag(double(sv));
    deno=(-C*M*DD*ttemp'+C*w*w'*A_new*ttemp*DD*ttemp');

    % adjust learning rate
    if l_new >  l_old
        eta = eta*0.9;
        if eta <1e-50
            break;
        else
            continue;
        end
    else
        eta = eta*1.1;
    end
%     if mod(cnt,1)==0
%         display(num2str([ cnt l_new  AKA  norm(A-A_new,'fro')/norm(A, 'fro') eta ]))
% %         plot(D); drawnow; pause(0.1)
%     end
    
    if l_old - l_new < eps && norm(A-A_new, 'fro')/norm(A, 'fro')<eps
        break;
    end

    l_old = l_new;
    A = A_new;
    AKA = AKA_new;
    cnt =cnt+1;
end
% display(num2str([ cnt l_old  AKA norm(A-A_new,'fro')/norm(A, 'fro') eta ]));
%% save the algorithm information and trained projection matrix.
if option.lambda>0
    Method.name = 'rPCCA';
else
    Method.name = 'PCCA';
end
% compute the metric loss
l_met=sum(logistic_loss(D,y,option))+ option.lambda * AKA_new;
Method.P=A_new;
Method.kernel=option.kernel;
% Method.Prob = [];
% Method.Dataname = option.dataname;
% Method.Ranking = [];
% Method.Dist = [];
Method.l_new=l_new;
%  digits(20);b=vpa(l_new) 
% Method.Trainoption=option;
Method.l_met=l_met;
return;

% Computing the generalized logistic loss value, the objective function 
% value eq(2). 
function L =logistic_loss(D,y,option)
beta = option.beta;
L =1/beta*log(1+ exp(beta*(y'.*(D-1))));
return;
