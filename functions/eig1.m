function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
% isMax=1，则选取A的最大特征值对应的特征向量
% isMax=0，则选取A的最小特征值对应的特征向量

if nargin < 2
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end;

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
end;
% [v d] = eig(A);

% -------------------- 方案1 -------------------------- %
% % 具体修改办法参照 http://ask.cvxr.com/t/eig-did-not-converge-in-prox-trace/996/16
try
    [v,d] = eig(A);
catch ME
    if (strcmpi(ME.identifier,'MATLAB:eig:NoConvergence'))
        [v,d] = eig(A, eye(size(A)));
    else
        rethrow(ME);
    end
end

% % % --------------------- 方案2 用SVD替换eig----------------------- %
% % [v,d,w] = svd(A);
% % d = d.*sign(diag(real(dot(v,w,1))));

d = diag(d);
%d = real(d);
if isMax == 0
    [d1, idx] = sort(d);                % 升序
else
    [d1, idx] = sort(d,'descend');      % 降序
end;    

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx);