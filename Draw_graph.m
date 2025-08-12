
% 
[maxacc,I_maxacc] = max(libACC(:))
[ind1, ind2, ind3, ind4] = ind2sub(size(libACC), I_maxacc)


% [maxacc,I_maxacc] = max(libACC(:))
% maxsen = libSEN(find(libACC==maxacc))
% maxspe = libSPE(find(libACC == maxacc))
% maxauc = libAUC(find(libACC == maxacc))
% maxaccstd = libaccstd(find(libACC == maxacc))
% maxsenstd = libsenstd(find(libACC == maxacc))
% maxspestd = libspestd(find(libACC == maxacc))
% maxaucstd = libaucstd(find(libACC == maxacc))
lamda_vector = [10^-5,10^-4,10^-3,10^-2,10^-1,1,10];
lamda_vector_2 = [10^-5,10^-4,10^-3,10^-2,10^-1,1,10];
lamda_vector_3 = [10^-5,10^-4,10^-3,10^-2,10^-1,1,10];
lamda_vector_4 = [10^-5,10^-4,10^-3,10^-2,10^-1,1,10];
B = permute(libACC, [3,4,1, 2]);
Draw_X = log10(lamda_vector);
Draw_Y = log10(lamda_vector_2);
ACC = B(:,:,2,2);
% ACC(2,1) = 60;

surf(Draw_X,Draw_Y,ACC);
xlabel('$\log(\lambda_3)$', 'Interpreter', 'latex','FontSize', 15,'FontWeight','bold')
ylabel('$\log(\lambda_4)$', 'Interpreter', 'latex','FontSize', 15,'FontWeight','bold')
zlabel('ACC','FontSize', 15)

% 设置 color bar 的显示范围
caxis([62.45, 80.29]);

% 添加 color bar，并设置刻度和标签
ticks = linspace(61.00, 81.00, 11);
tickLabels = arrayfun(@(x) sprintf('%.2f', x), ticks, 'UniformOutput', false);
colorbar('Ticks', ticks, 'TickLabels', tickLabels);

% max(libAUC(:))

% PCGSXROC = load('MyMethodXROCPCGS.mat');
% PCGSYROC = load('MyMethodYROCPCGS.mat');
% PCGSXROC = PCGSXROC.XROC_ALL;
% PCGSYROC = PCGSYROC.YROC_ALL;
% 
% 
% DMFTXROC = load('MyMethodXROCDMFT.mat');
% DMFTYROC = load('MyMethodYROCDMFT.mat');
% DMFTXROC = DMFTXROC.XROC_ALL;
% DMFTYROC = DMFTYROC.YROC_ALL;
% 
% MYXROC = load('MyMethodXROC2.mat');
% MYYROC = load('MyMethodYROC2.mat');
% MYXROC = MYXROC.libXROC;
% MYYROC = MYYROC.libYROC;
% 
% 
% PCGSXROC(16,:) = 1;
% PCGSYROC(16,:) = 1;
% DMFTXROC(16,:) = 1;
% DMFTYROC(16,:) = 1;
% MYXROC(16,:) = 1;
% MYYROC(16,:) = 1;
% 
% theauc = trapz(MYXROC(1:16,:),MYYROC(1:16,:))
% 
% A_as_column1 = MyMethodlibXROC(1:16,:);
% A_as_column2 = MyMethodlibYROC(1:16,:);
% 
% B_as_column1 = A_as_column1(:);
% B_as_column2 = A_as_column2(:);
%  hold on;
% plot(B_as_column1,B_as_column2,'Color',[0.6350 0.0780 0.1840],'linewidth',2.5);


plot(MyMethodlibXROC(1:16,5),MyMethodlibYROC(1:16,5),'Color',[0.6350 0.0780 0.1840],'linewidth',2.5);
hold on;
plot(PCGSlibXROC(1:16,:),PCGSlibYROC(1:16,:),'r-','linewidth',2.5,'LineStyle', '--');
hold on;
plot(DMTFSlibXROC(1:16,:),DMTFSlibYROC(1:16,:),'g-','linewidth',2.5,'LineStyle', '--');
hold on;
plot(HYPERlibXROC(1:16,:),HYPERlibYROC(1:16,:),'b-','linewidth',2.5,'LineStyle', '--');
hold on;
plot(LATENT2022Xsvm(1:16,:),LATENT2022Ysvm(1:16,:),'c-','linewidth',2.5,'LineStyle', '--');
hold on;
plot(LATENTTauXsvm(1:16,:),LATENTTauYsvm(1:16,:),'m-','linewidth',2.5,'LineStyle', '--');
hold on;
plot(LOWRANKlibXROC(1:17,:),LOWRANKlibYROC(1:17,:),'y-','linewidth',2.5,'LineStyle', '--');
hold on;
plot(M2TFSlibXROC(1:16,:),M2TFSlibYROC(1:16,:),'Color',[0.4660 0.6740 0.1880],'linewidth',2.5,'LineStyle', '--');
hold on;
title('pMCI vs. sMCI');
xlabel('False positive rate(FPR)');ylabel('True positive rate(TPR)');
legend('Ours','PCGS','DMTFS','Hypergraph','LFRL','LRL','Low-rank','MTFS');
% 
% 
% imagesc(L_pet)
% axis square
% title('PET similarity matrix')
% colorbar
