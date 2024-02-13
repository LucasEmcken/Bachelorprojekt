   % Update T 
   P.Sf = krprod(krprt,Sf.').';
   P.f = krprod(ones(size(krprt,1),1),f.').';
   P.w = ones(1,length(f));
   if mod(N(2),2) == 0
       P.w(2:end-1) = 2;
   else
       P.w(2:end) = 2;
   end
   P.w = krprod(ones(size(krprt,1),1),P.w')';
   P.sizeX2 = size(X,2);       
   P.At = FACT{1};
   P.Xft = matricizing(Xf,1);
   for m = 1:size(Xf,1)
       if estWholeSamples == 1
           [T(m,:),FACT{1}(m,:)] = estTimeAutCor(P.Xft(m,:),P.At(m,:),Sf,krprt,P.Sf,P.f,T(m,:),Nf,N,P.w,constr(1),TauWMatrix,sigma_sq*Lambda);
       else
           P.A = P.At(m,:);
           P.Xf = P.Xft(m,:);
           P.nyT = nyT(m);
           [T(m,:),nyT(m)] = update_T(T(m,:),P,TauW);           
       end       
   end

      % Center Delays
   for d=1:noc
      if sum(TauWMatrix(d,:))==N(2)
       tmean = round(mean(T(:,d)));
       T(:,d) = T(:,d)-tmean;
       Sf(d,:) = Sf(d,:).*exp(tmean*f);      
      end
   end

   function [T,nyT,cost] = update_T(T,P,TauW)
   nyT = P.nyT;
   Sf = P.Sf;
   A = P.A;
   sizeX2 = P.sizeX2;
   Xf = P.Xf;
   f = P.f;
   w = P.w;  
   Recfd = zeros(size(A,1),size(Sf,2),size(A,2));
   for d = 1:size(A,2)
       Recfd(:,:,d) = (repmat(A(:,d),[1 length(f)]).*exp(T(:,d)*f)).*repmat(Sf(d,:),[size(A,1),1]);
   end
   Recf = sum(Recfd,3);
   Q = Recfd.*repmat(conj(Xf-Recf),[1,1,size(Recfd,3)]);
   grad = squeeze(sum(repmat((w.*f),[size(Q,1),1,size(Q,3)]).*(conj(Q)-Q),2))';  
   ind1 = find(w == 2); % Areas used twice
   ind2 = find(w == 1); % Areas used once
   cost_old = norm(Xf(:,ind1)-Recf(:,ind1),'fro')^2;
   cost_old = cost_old+0.5*norm(Xf(:,ind2)-Recf(:,ind2),'fro')^2;
   keepgoing = 1;
   Told = T;
   while keepgoing
       T = Told-nyT*grad;
       for d=1:length(T)
          if T(d)<TauW(d,1)
              T(d)=TauW(d,1);
          end
          if T(d)>TauW(d,2)
              T(d)=TauW(d,2);
          end
       end
       for d = 1:size(A,2)
           Recfd(:,:,d) = (repmat(A(:,d),[1 length(f)]).*exp(T(:,d)*f)).*repmat(Sf(d,:),[size(A,1),1]);
       end
       Recf = sum(Recfd,3);
       cost = norm(Xf(:,ind1)-Recf(:,ind1),'fro')^2;
       cost = cost+0.5*norm(Xf(:,ind2)-Recf(:,ind2),'fro')^2;
       if cost<= cost_old
           keepgoing = 0;
           nyT = nyT*1.2;
       else
           keepgoing = 1;
           nyT = nyT/2;
       end
   end
   T = mod(T,sizeX2);
   ind = find(T>floor(sizeX2/2));
   T(ind) = T(ind)-sizeX2;