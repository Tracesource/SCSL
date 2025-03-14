function [S,G,iter,obj] = missingalgo_qp_each_1(X,Y,beta,lambda,ind,index,ini)

%% initialize
maxIter = 50 ; % the number of iterations

numclass = length(unique(Y));
k = numclass;
numview = length(X);
numsample = size(Y,1);

missingindex = constructA(ind);

%% Calculate Omega
Omega = 0; 
for iv = 1:numview
    for jv = 1:numview
        Omega = Omega+ind(:,iv)*ind(:,jv)';
    end
end
Omega = 1./Omega;
clear ind_plus

%% Initialize G with k-means
eff_g = ind*ones(1,numview)';
eff_g = 1./eff_g;

rng(12,'twister');
G = 0;

if ini == 1
    for iv = 1:numview  
        rng(12,'twister');
        if size(X{iv},1)<k
            break;
        end
        [~,CC] = kmeans(X{iv},k, 'MaxIter',100,'Replicates',10);
        G = G+CC.*(eff_g*ones(1,k))';
    end
end
if ini == 2
    for iv = 1:numview  
        options = [];
        options.ReducedDim = k;
        [CC,~] = PCA1(X{iv}, options);
        if size(X{iv},1)<k
            break;
        end
        G = G+CC'.*(eff_g*ones(1,k))';
    end
end
clear CC

%% Initialize S with intra-view distance
S = zeros(numsample,numsample); 
for iv = 1:numview
    ind_0 = find(ind(:,iv) == 0);
    H = diag(ind(:,iv));
    H(:,ind_0) = [];
    numsample1 = size(index{iv},1);
    s = zeros(numsample1,numsample1); 
    for jj=1:numsample1
        a = zeros(1,numsample1);
        b = zeros(1,numsample1);
        for ii =1:numsample1
            a(ii) = norm(X{iv}(:,index{iv}(ii))-X{iv}(:,index{iv}(jj)),'fro')^2;
            b(ii) = norm(G(:,index{iv}(ii))-G(:,index{iv}(jj)),'fro')^2;
        end
        [s(:,jj),~] = EProjSimplex_new(2*a,lambda/2*b);
    end
    S = S+(H*s*H').*(eff_g*ones(1,numsample))';
end
clear a b eff_g s H ind_0

%% Initialize W
for iv = 1:numview
    pp = X{iv}*G';
    [Unew,~,Vnew] = svd(pp,'econ');
    W{iv} = Unew*Vnew';
end

%%
S1 = (S+S')/2;
Sum_S = sum(S1);
L = diag(Sum_S)-S1;

%%
flag = 1;
iter = 0;

while flag
    iter = iter + 1;
    
   %% optimize W
    Y = Omega.*S.*S;
    for iv = 1:numview
        D = 0;
        Z = 0;
        for jv = 1:numview
            Z1 = zeros(numsample,numsample);
            for ii = 1:numsample
                Z1(ii,ii)=Y(ii,:)*ind(:,jv);
            end
            Z = Z+X{iv}*Z1*X{iv}';
            D = D+X{iv}*Y*X{jv}'*W{jv};
        end
        D = D+beta*X{iv}*G';
        W{iv} = OrthQP_solver(2*Z,-2*D);
    end

    %% optimize G
    pg = 0;
    ag = 0;
    for iv = 1:numview
        pg = pg+W{iv}'*X{iv};
        ag = ag+diag(missingindex{iv});
    end
    G = pg/(lambda/beta*L+ag);
%     G = pg/(lambda/beta*L+numview*eye(numsample));
    
    %% optimize S
    GG = G'*G;
    LG = diag(GG)*repmat(1,numsample,1)';
    B = LG+LG'-2*GG;
    
    A = 0;
    for iv = 1:numview
        WX{iv} = W{iv}'*X{iv};
        XX{iv} = diag(WX{iv}'*WX{iv})*repmat(1,numsample,1)';
    end
    for iv = 1:numview
        for jv = 1:numview
            A = A+XX{iv}.*repmat(missingindex{jv},numsample,1)+(XX{jv}.*repmat(missingindex{iv},numsample,1))'-2*WX{jv}'*WX{iv};
        end
    end
    A = A.*Omega;
    parfor jj=1:numsample
        [S(:,jj),~] = EProjSimplex_new(2*A(:,jj),lambda/2*B(:,jj));
    end
    
    %%
    A = S.*S.*A;
    term1 = sum(sum(A));

    term2 = 0;
    for iv = 1:numview
        term2 = term2 + norm(X{iv} - W{iv} * (G.*repmat(missingindex{iv},k,1)),'fro')^2;
    end

    S1 = (S+S')/2;
    Sum_S = sum(S1);
    L = diag(Sum_S)-S1;
    term3 = trace(G*L*G');

    obj(iter) = term1+beta*term2+lambda*term3;
    
    if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        flag = 0;
    end
end
