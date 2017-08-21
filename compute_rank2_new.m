% calculate the matching characteristics.
% By Fei Xiong, 
%    ECE Dept, 
%    Northeastern University 
%    2013-11-04
% Input: 
%       Method: the distance learning algorithm struct. In this function
%       two field are used. 
%               P is the projection matrix. d'-by-Ntr (assume the kernel trick is used.)
%               kernel is the name of the kernel function. 
%       train: The data used to learn the projection matric. Each row is a
%               sample vector. Ntr-by-d
%       test: The data used to test and calculate the CMC for the
%               algorithm. Each row is a sample vector. Nts-by-d
%       ix_partition: the randomly generated partition of the test set.
%               Each row is a randomly generated partition. 1 represents
%               this test sample is used as reference sample, while 0
%               represents such sample is used as probe sample. Nit-by-Nts
%       IDs: The identity of the samples in the test set. Nts-by-1, where
%               Nts is the size of test set. Nts-by-1
function [K_test] = compute_rank2_new(Method, train, test)


% for k =1:size(ix_partition,1) % calculate the CMC for each random partition.
    % set the kernel matrix for reference and prob set.
    % ix_ref = ix_partition(k,:) ==1;
    % when the probe set is not the same as test gallery set, it will be
    % labeled as "-1"
    % if min(min(double(ix_partition))) < 0
    %    ix_prob = ix_partition(k,:) ==-1; 
    % else
    %    ix_prob = ix_partition(k,:) ==0;
    % end
    % ref_ID = IDs(ix_ref);
    % prob_ID = IDs(ix_prob);
%     [rows,cols]=size(test);
%     dis = zeros(rows,rows);
 %   for c = 1:numel(test)
%  for kk=1:2
%        A = Method.P; % Projection vector
       % if strcmp(Method{c}.name,'oLFDA')
       %    K_test = test{c}';
       % else
       [K_test] = ComputeKernelTest(train, test, Method); %compute the kernel matrix.
       
       % end
%         K_ref = K_test;
%         K_prob = K_test;
%         for i =1: size(K_prob,2)
%             diff = bsxfun(@minus, K_ref,K_prob(:,i));
%             diff = A*diff;
%             dis(i, :) = dis(i, :) + sum(diff.^2,1);
%         end
 %  end
 % calculate the distance and ranking for each prob sample
%    for i =1:sum(ix_prob)
%         diff = bsxfun(@minus, K_ref,K_prob(:,i));
%         diff = A*diff;
%         dis(i, :) = sum(diff.^2,1);
%         [tmp, ix] = sort(dis(i, :));
%         r(i) =  find(ref_ID(ix) == prob_ID(i));
%         ixx(i,:)=ix;
%    end
    % returned ranking matrix, each row is the ranking for a reference/prob
    % set partition
    % R(k, :) = r; 
%     Alldist(:,:,kk)= dis; % distance matrix
%  end
% end
return;

% Calculate the kernel matrix for train and test set.
% TODO: Replace the ComputeKernel function in  ComputeKernel.m
% Input: 
%       Method: the distance learning algorithm struct. In this function
%               only field used "kernel", the name of the kernel function. 
%       train: The data used to learn the projection matric. Each row is a
%               sample vector. Ntr-by-d
%       test: The data used to test and calculate the CMC for the
%               algorithm. Each row is a sample vector. Nts-by-d
function [K_test] = ComputeKernelTest(train, test, Method)

if (size(train,2))>2e4 && (strcmp(Method.kernel, 'chi2') || strcmp(Method.kernel, 'chi2-rbf'))
    % if the input data matrix is too large then use parallel computing
    % tool box.
    matlabpool open
    
    switch Method.kernel
        case {'linear'}
            K_test = train * test';
        case {'chi2'}
            parfor i =1:size(test,1)
                dotp = bsxfun(@times, test(i,:), train);
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) = 2* sum(dotp./(sump+1e-10),2);
            end
        case {'chi2-rbf'}
            sigma = Method.rbf_sigma;
            parfor i =1:size(test,1)
                subp = bsxfun(@minus, test(i,:), train);
                subp = subp.^2;
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) =  sum(subp./(sump+1e-10),2);
            end
            K_test =exp(-K_test./sigma);
    end
    matlabpool close
else
    switch Method.kernel
        case {'linear'}
            K_test = train * test';
        case {'chi2'}
            for i =1:size(test,1)
                dotp = bsxfun(@times, test(i,:), train);
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) = 2* sum(dotp./(sump+1e-10),2);
            end
        case {'chi2-rbf'}
            sigma = Method.rbf_sigma;
            for i =1:size(test,1)
                subp = bsxfun(@minus, test(i,:), train);
                subp = subp.^2;
                sump = bsxfun(@plus, test(i,:), train);
                K_test(:,i) =  sum(subp./(sump+1e-10),2);
            end
            K_test =exp(-K_test./sigma);
        case {'gaus-rbf'}
             %% Gaussian RBF kernel: myself
            sigma = Method.rbf_sigma;
            ro_tr=size(train,1);          
            for i = 1:ro_tr           
                subp=bsxfun(@minus,train(i,:),test)';
                K_test(i,:)= arrayfun(@(x) norm(subp(:,x)),1:size(subp,2));     
            end
            K_test=exp(-K_test./sigma);
    end
end
return;