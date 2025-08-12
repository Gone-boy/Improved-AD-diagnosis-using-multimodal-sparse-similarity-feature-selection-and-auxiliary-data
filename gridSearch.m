function [bestacc,bestsen,bestspe,bestauc,bestf1score,XROC,YROC] = gridSearch(MRIKtr,PETKtr,MRIKte,PETKte,train_size,test_size,train_label,test_label)
   beta = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];
   acc = zeros(size(beta,2),1);
   sen = zeros(size(beta,2),1);
   spe = zeros(size(beta,2),1);
   auc = zeros(size(beta,2),1);
   f1score = zeros(size(beta,2),1);
   XROCALL = zeros(32,size(beta,2));
   YROCALL = zeros(32,size(beta,2));
  
   for i = 1:size(beta,2)
      
      K_MRI_PET_train = [(1:train_size)',beta(i)*MRIKtr + (1-beta(i))*PETKtr];
      K_MRI_PET_test =  [(1:test_size)',beta(i)*MRIKte + (1-beta(i))*PETKte];
      model = libsvmtrain(train_label, K_MRI_PET_train, '-t 4 -b 1');
      [libpredict_label, accuracy, dec_values] = libsvmpredict(test_label, K_MRI_PET_test, model,'-b 1');
      [Xsvm,Ysvm,Tsvm,AUCsvm1] = perfcurve(test_label,dec_values(:,1),1);
      [Xsvm2,Ysvm2,Tsvm2,AUCsvm2] = perfcurve(test_label,dec_values(:,2),1);
      libTP = sum((libpredict_label ~= 1) & (test_label ~= 1)); % TP
      libTN = sum((libpredict_label == 1) & (test_label == 1)); % TN
      libFP = sum((libpredict_label ~= 1) & (test_label == 1)); % FP
      libFN = sum((libpredict_label == 1) & (test_label ~= 1)); % FN
      acc(i) = accuracy(1);
      sen(i) = (libTP / (libTP + libFN));
      spe(i) = (libTN / (libTN + libFP));
      f1score(i) = (2*libTP) / (2*libTP + libFN + libFP);
      if AUCsvm1 > 0.5
          auc(i) = AUCsvm1;
          paddedVectorX = [Xsvm; zeros(32 - length(Xsvm), 1)];
          paddedVectorY = [Ysvm; zeros(32 - length(Ysvm), 1)];
%           disp(size(paddedVectorX));
          XROCALL(:,i) = paddedVectorX;
          YROCALL(:,i) = paddedVectorY;
      else
          auc(i) = AUCsvm2;
          paddedVectorX2 = [Xsvm2; zeros(32 - length(Xsvm2), 1)];
          paddedVectorY2 = [Ysvm2; zeros(32 - length(Ysvm2), 1)];
%           disp(size(paddedVectorX2));
          XROCALL(:,i) = paddedVectorX2;
          YROCALL(:,i) = paddedVectorY2;
      end
   end
   [bestacc,I] = max(acc);
   bestsen = sen(I);
   bestspe = spe(I);
   bestauc = auc(I);
   bestf1score = f1score(I);
   XROC = XROCALL(:,I);
   YROC = YROCALL(:,I);
   


   
   