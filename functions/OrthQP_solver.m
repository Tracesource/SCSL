function H = OrthQP_solver(G,B,H)
% ****************** Introduction **********************
% Problem: max_H 1/2*trace(H'*G*H)+trace(H'*B),        *
%          s.t. H'*H=I.                                *
% G: n*n, Symmetric matrix                             *
% B: n*k, n>k                                          *
% H: n*k, Optional input                               *
% ******************************************************
% rep：迭代次数                                         *
% fobj：目标函数值                                      *
% 包含 3种初始化方式 & 加速，全局修正                    *
% ******************************************************

[n,k] = size(B);
% G = G - eye(n) * (eigs(G, 1, 'smallestreal')-1e-5);
G = G - eye(n) * (0-1/n*1e-5);

N1 = 100;% 最大迭代次数
fobj = 1:N1;% 目标函数值

% ****************** 3种初始化方式 **********************
if nargin < 3
    % 1不好的初始化
    % [H,~,~] = -svds(B,k);
    % 2随机初始化
    % [H,~,~] = svds(rand(n,k),k);
    % 3我的初始化
    ii = find(isnan(B));
    B(ii) = zeros(size(ii));
    [U,~,V] = svds(B,k);
    H = U*V';
end

for rep = 1:N1
    M = G * H + B;
    ii = find(isnan(M));
    M(ii) = zeros(size(ii));
    [U,~,V] = svds(M,k);
    H = U*V';
    
    % *************** 加速，全局修正 ******************
    CC=H'*B;
    ii = find(isnan(CC));
    CC(ii) = zeros(size(ii));
    [U,~,V] = svds(CC,k);
    H=H*U*V';
    
    fobj(rep) = 0.5*trace(H'*G*H)+trace(B'*H);
    if rep>4 && ((fobj(rep)-fobj(rep-1))/(fobj(2)-fobj(1))<1e-3)
        break;
    end
end
end
