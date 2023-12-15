function H = OrthQP_solver(G,B,H)
% ****************** Introduction **********************
% Problem: max_H 1/2*trace(H'*G*H)+trace(H'*B),        *
%          s.t. H'*H=I.                                *
% G: n*n, Symmetric matrix                             *
% B: n*k, n>k                                          *
% H: n*k, Optional input                               *
% ******************************************************
% rep����������                                         *
% fobj��Ŀ�꺯��ֵ                                      *
% ���� 3�ֳ�ʼ����ʽ & ���٣�ȫ������                    *
% ******************************************************

[n,k] = size(B);
% G = G - eye(n) * (eigs(G, 1, 'smallestreal')-1e-5);
G = G - eye(n) * (0-1/n*1e-5);

N1 = 100;% ����������
fobj = 1:N1;% Ŀ�꺯��ֵ

% ****************** 3�ֳ�ʼ����ʽ **********************
if nargin < 3
    % 1���õĳ�ʼ��
    % [H,~,~] = -svds(B,k);
    % 2�����ʼ��
    % [H,~,~] = svds(rand(n,k),k);
    % 3�ҵĳ�ʼ��
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
    
    % *************** ���٣�ȫ������ ******************
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
