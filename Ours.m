
% [maxacc,I_maxacc] = max(libACC(:))
% [ind1, ind2, ind3, ind4] = ind2sub(size(libACC), I_maxacc)
% % max(ACC(:))
% maxsen = libSEN(find(libACC==maxacc))
% maxspe = libSPE(find(libACC == maxacc))
% maxauc = libAUC(find(libACC == maxacc))
% maxaccstd = libaccstd(find(libACC == maxacc))
% maxsenstd = libsenstd(find(libACC == maxacc))
% maxspestd = libspestd(find(libACC == maxacc))
% maxf1score = libF1score(find(libACC == maxacc))



% dataprocess;
lamda_vector = [10^-5,10^-4,10^-3,10^-2,10^-1,1,10,100,1000];
lamda_vector_2 = [10^-5,10^-4,10^-3,10^-2,10^-1,1,10,100,1000];
lamda_vector_3 = [10^-5,10^-4,10^-3,10^-2,10^-1,1,10,100,1000];
lamda_vector_4 = [10^-5,10^-4,10^-3,10^-2,10^-1,1,10,100,1000];

% MCI_AD_dPET_LABEL = MCI_AD_dPET_label - 1;


cross_validata_acc = 0;
cross_validata_sen = 0;
cross_validata_spe = 0;



libcross_validata_acc = 0;
libcross_validata_sen = 0;
libcross_validata_spe = 0;
libcross_validata_auc = 0;
libcross_validata_f1score = 0;
% EMCI_LMCI_label = double(EMCI_LMCI_label);
% NC_AD_label = double(NC_AD_label);
XROC_ALL = zeros(35,10);
YROC_ALL = zeros(35,9);
MyMethody_trueall = zeros(16,10);
MyMethody_predall = zeros(16,10);


rng(42);
% indices = cvpartition(size(MCI_NC_mri_data_class, 1), 'KFold', 10); % Perform 10-fold cross-validation and obtain the indices of each fold in the dataset.

MRIROI_array = zeros(30,10);
PETROI_array = zeros(30,10);



for i = 4:7
    for j = 4:7
        for k = 4:7
            for l = 4:7
               MCI_NC_mri_rammdon = [pMCI_sMCI_andPETMRIdata pMCI_sMCI_PETdata pMCI_sMCI_andPET_label];
               NC_AD_ramdon = [NC_AD_mri_pet NC_AD_pet NC_AD_pet_label];
                % Combine MRI and PET modalities along with their labels
         
                
               minorityClass = MCI_NC_mri_rammdon(MCI_NC_mri_rammdon(:, end) == 1, :);
               majorityClass = MCI_NC_mri_rammdon(MCI_NC_mri_rammdon(:, end) == 0, :);


               au_minorityClass = NC_AD_ramdon(NC_AD_ramdon(:,end) == 2,:);
               au_majorityClass = NC_AD_ramdon(NC_AD_ramdon(:,end) == 0,:);
    
                % Calculate the number of samples in the minority class
               minorityCount = size(minorityClass, 1);
               au_minorityCount = size(au_minorityClass,1);
    
                %  random sample
               randIndices = randperm(size(majorityClass, 1), minorityCount);
               selectedMajority = majorityClass(randIndices, :);
               
               au_randIndices = randperm(size(au_majorityClass, 1), au_minorityCount);
               au_selectedMajority = au_majorityClass(au_randIndices, :);

    
               MCI_NC_mri_data_class = [minorityClass(:,1:93);selectedMajority(:,1:93)];
               MCI_NC_pet_data_class = [minorityClass(:,94:end-1);selectedMajority(:,94:end-1)];
               MCI_NC_label_data_class = [minorityClass(:,end);selectedMajority(:,end)];
%                MCI_NC_label_data_class(MCI_NC_label_data_class == 2) = 1;

               au_MCI_NC_mri_data_class = [au_minorityClass(:,1:93);au_selectedMajority(:,1:93)];
               au_MCI_NC_pet_data_class = [au_minorityClass(:,94:end-1);au_selectedMajority(:,94:end-1)];
               au_MCI_NC_label_data_class = [au_minorityClass(:,end);au_selectedMajority(:,end)];
               
%                au_MCI_NC_mri_data_class = [au_selectedMajority(:,1:93);minorityClass(:,1:93)];
%                au_MCI_NC_pet_data_class = [au_selectedMajority(:,94:end-1);minorityClass(:,94:end-1)];
%                au_MCI_NC_label_data_class = [au_selectedMajority(:,end);minorityClass(:,end)];

%                au_MCI_NC_mri_data_class = [au_selectedMajority(:,1:93);minorityClass(:,1:93)];
%                au_MCI_NC_pet_data_class = [au_selectedMajority(:,94:end-1);minorityClass(:,94:end-1)];
%                au_MCI_NC_label_data_class = [au_selectedMajority(:,end);minorityClass(:,end)];
                


%                perm = randperm(size(MCI_NC_mri_data_class,1));
%                MCI_NC_mri_data_class = MCI_NC_mri_data_class(perm, :);
%                MCI_NC_pet_data_class = MCI_NC_pet_data_class(perm, :);
%                MCI_NC_label_data_class = MCI_NC_label_data_class(perm);
% 
%                au_perm = randperm(size(au_MCI_NC_mri_data_class,1));
%                au_MCI_NC_mri_data_class = au_MCI_NC_mri_data_class(au_perm, :);
%                au_MCI_NC_pet_data_class = au_MCI_NC_pet_data_class(au_perm, :);
%                au_MCI_NC_label_data_class = au_MCI_NC_label_data_class(au_perm);

               indices = cvpartition(size(MCI_NC_mri_data_class, 1), 'KFold', 10); 
               for c_v = 1:10
                    test = indices.test(c_v); % split test set
                    train = indices.training(c_v); % split train set
                    % training set
                    X_new_train = MCI_NC_mri_data_class(train,:);
                    X_new_train_pet = MCI_NC_pet_data_class(train,:);

                    Y_new_train = MCI_NC_label_data_class(train);
                    X_new_train_mri_a = au_MCI_NC_mri_data_class(train,:);
                    X_new_trainpet_a = au_MCI_NC_pet_data_class(train,:);
                    Y_new_train_label_a = au_MCI_NC_label_data_class(train);
                    % test set
                    X_new_test = MCI_NC_mri_data_class(test,:);
                    X_new_test_pet = MCI_NC_pet_data_class(test,:);
                    Y_new_test = MCI_NC_label_data_class(test);
                    X_new_test_mri_a = au_MCI_NC_mri_data_class(test,:);
                    X_new_testpet_a = au_MCI_NC_pet_data_class(test,:);
                    Y_new_test_label_a = au_MCI_NC_label_data_class(test);
                    % feature selection

                    obji = 1;
                    W_t = zeros(93,2);
                    W_a = zeros(93,2);
                    W_tpet = zeros(93,2);
                    W_apet = zeros(93,2);
                    iter = 1;
                    X_t = X_new_train;
                    X_tpet = X_new_train_pet;
                    Y_ta = Y_new_train;
                    % for MCI vs AD
%                     Y_ta = Y_ta - 1;
                    Y_ta(Y_ta == 2) = 1;
                    Y_t = [Y_ta ones(size(Y_ta,1),1)-Y_ta];
                    X_a = X_new_train_mri_a;
                    X_apet = X_new_trainpet_a;
                    Y_at = Y_new_train_label_a;
                    Y_at(Y_at ==2) = 1;
                    Y_a = [Y_at ones(size(Y_at,1),1)-Y_at];
                    % train random forest
                    Mdl_mri = TreeBagger(93,X_t,Y_t(:,1),...
                                        'Method',"classification",...
                                        'OOBPrediction',"on");
                    Mdl_mri = fillprox(Mdl_mri);
                    L_mri = Mdl_mri.Proximity;
    
                    Mdl_pet = TreeBagger(93,X_tpet,Y_t(:,1),...
                        'Method',"classification",...
                        'OOBPrediction',"on");
                    Mdl_pet = fillprox(Mdl_pet);
                    L_pet = Mdl_pet.Proximity;
                    
%                     L_all = zeros(size(X_t,1),size(X_t,1));
%                     for i_g = 1:size(X_t,1)
%                         for j_g = 1:size(X_t,1)
%                             if L_mri(i_g,j_g) ~=0 && L_pet(i_g,j_g) ~= 0
%                                     L_all(i_g,j_g) = (L_mri(i_g,j_g) + L_pet(i_g,j_g)) / 2;
%                             end
%                         end
%                     end
                    
                      while 1
                          
                          W1 = sqrt(sum([W_t W_a].*[W_t W_a], 2)+ eps); % caculate the W_1's l2,1 norm
                          w1 = 0.5./W1;
                          U = diag(w1);
    
                          W1_pet = sqrt(sum([W_tpet W_apet].*[W_tpet W_apet], 2)+ eps); % caculate the W_1's l2,1 norm
                          w1_pet = 0.5./W1_pet;
                          U_pet = diag(w1_pet);
    
        
                          W2 = sqrt(sum((Y_a - X_a*W_a).*(Y_a - X_a*W_a),2) + eps); %caculate the |X_m*W_m - X_p*W_p|'s l2,1 norm
                          w2 = 0.5./W2;
                          B = diag(w2);
    
                          W2_pet = sqrt(sum((Y_a - X_apet*W_apet).*(Y_a - X_apet*W_apet),2) + eps); %caculate the |X_m*W_m - X_p*W_p|'s l2,1 norm
                          w2_pet = 0.5./W2_pet;
                          B_pet = diag(w2_pet);
                          
    
                          I_d = eye(93);
                %           Update W_1
                          W_t = (X_t'*X_t + lamda_vector_2(j)*U + lamda_vector_3(k)*I_d + lamda_vector_4(l)*X_t'*L_mri*X_t)\(X_t'*Y_t);
      
                %           Update W_p
                          W_a = (lamda_vector(i)*X_a'*B*X_a + lamda_vector_2(j)*U + lamda_vector_3(k)*I_d)\(lamda_vector(i)*X_a'*B*Y_a);
    
                          % Update W_tpet
                          W_tpet = (X_tpet'*X_tpet + lamda_vector_2(j)*U_pet + lamda_vector_3(k)*I_d + lamda_vector_4(l)*X_tpet'*L_pet*X_tpet)\(X_tpet'*Y_t);
                          % Update W_apet
                          W_apet = (lamda_vector(i)*X_apet'*B_pet*X_apet + lamda_vector_2(j)*U_pet + lamda_vector_3(k)*I_d)\(lamda_vector(i)*X_apet'*B_pet*Y_a);
                
                          W1 = sqrt(sum([W_t W_a].*[W_t W_a], 2)+ eps); % caculate the W_1's l2,1 norm
                          w1 = 0.5./W1;
                          U = diag(w1);
    
                          W1_pet = sqrt(sum([W_tpet W_apet].*[W_tpet W_apet], 2)+ eps); % caculate the W_1's l2,1 norm
                          w1_pet = 0.5./W1_pet;
                          U_pet = diag(w1_pet);
    
        
                          W2 = sqrt(sum((Y_a - X_a*W_a).*(Y_a - X_a*W_a),2) + eps); %caculate the |Y_a - X_a*W_a|'s l2,1 norm
                          w2 = 0.5./W2;
                          B = diag(w2);
    
                          W2_pet = sqrt(sum((Y_a - X_apet*W_apet).*(Y_a - X_apet*W_apet),2) + eps); %caculate the |X_m*W_m - X_p*W_p|'s l2,1 norm
                          w2_pet = 0.5./W2_pet;
                          B_pet = diag(w2_pet);
                          
                          I_d = eye(93);
        
%                           obj = norm(Y_t - X_t*W_t,'fro')^2 + lamda_vector(i)*trace((Y_a - X_a*W_a)'*B*(Y_a - X_a*W_a)) + lamda_vector_2(j)*trace([W_t W_a]'*U*[W_t W_a]) + lamda_vector_3(k)*norm([W_t W_a],'fro')^2 + lamda_vector_4(l)*trace((X_t*W_t)'*L_mri*(X_t*W_t)) + ...
%                               norm(Y_t - X_tpet*W_tpet,'fro')^2 + lamda_vector(i)*trace((Y_a - X_apet*W_apet)'*B_pet*(Y_a - X_apet*W_apet)) + lamda_vector_2(j)*trace([W_tpet W_apet]'*U_pet*[W_tpet W_apet]) + lamda_vector_3(k)*norm([W_tpet W_apet],'fro')^2 + lamda_vector_4(l)*trace((X_tpet*W_tpet)'*L_pet*(X_tpet*W_tpet));
                          
                                  
                          obj(iter) = norm(Y_t - X_t*W_t,'fro')^2 + lamda_vector(i)*trace((Y_a - X_a*W_a)'*B*(Y_a - X_a*W_a)) + lamda_vector_2(j)*trace([W_t W_a]'*U*[W_t W_a]) + lamda_vector_3(k)*norm([W_t W_a],'fro')^2 + lamda_vector_4(l)*trace((X_t*W_t)'*L_mri*(X_t*W_t)) + ...
                              norm(Y_t - X_tpet*W_tpet,'fro')^2 + lamda_vector(i)*trace((Y_a - X_apet*W_apet)'*B_pet*(Y_a - X_apet*W_apet)) + lamda_vector_2(j)*trace([W_tpet W_apet]'*U_pet*[W_tpet W_apet]) + lamda_vector_3(k)*norm([W_tpet W_apet],'fro')^2 + lamda_vector_4(l)*trace((X_tpet*W_tpet)'*L_pet*(X_tpet*W_tpet));
                          cver = abs((obj(iter)-obji)/obji);
                          obji = obj(iter); 
                          iter = iter + 1;
                          
                          if (cver < 10^-6 && iter > 1) || iter > 2000,    break,     end
                      end
                    obj = obj / size(Y_t,1);
                    figure, plot(obj,'LineWidth',2);
                    title('Objective Function Curve','FontWeight', 'bold');
                    xlabel('Iterations','FontWeight', 'bold');
                    xlim([1, iter]);  
                    ylim([-inf, 2]);  
                    ylabel('Objective Function Value','FontWeight', 'bold');
                    obj = [];
                        % selected features
                    row_sum_abs = sum(abs(W_t), 2);
                    row_sum_abs_PET = sum(abs(W_tpet),2);
                       
                    [~, idx] = sort(row_sum_abs, 'descend');
                    [~,PETidx] = sort(row_sum_abs_PET,'descend');
                    
                    MRIROI_array(:,c_v) = idx(1:30);
                    PETROI_array(:,c_v) = PETidx(1:30);
    
                    
                   
                    count_positive = sum(MCI_NC_label_data_class == 1);
    
                    
                    count_negative = sum(MCI_NC_label_data_class == 0);
               
                   
                    X_train = X_new_train;
                    y_lasso_label =Y_new_train;
                    X_test = X_new_test;
                    y_lasso_latest = Y_new_test;
                    X_train = X_train(:,idx(1:30));
                    X_test = X_test(:,idx(1:30));
                   
                    X_train_pet = X_new_train_pet;
                    X_test_pet =X_new_test_pet;
                    
                    
                    X_train_pet = X_train_pet(:,PETidx(1:30));
                    X_test_pet = X_test_pet(:,PETidx(1:30));
                   

                    X_train_mrinor = normalize(X_train);
                    X_test_mrinor = normalize(X_test);
                    X_train_petnor = normalize(X_train_pet);
                    X_test_petnor = normalize(X_test_pet);
                    
                    MRIK_tr = X_train_mrinor*X_train_mrinor';
                    PETK_tr = X_train_petnor*X_train_petnor';
                    MRIK_te = X_test_mrinor*X_train_mrinor';
                    PETK_te = X_test_petnor*X_train_petnor';
                    
                    
%                     imagesc(PETK_tr)
%                     axis square
%                     title('PET similarity matrix')
%                     colorbar
                    
                    train_size = sum(train);
                    test_size = sum(test);
%                     [libacc,libsen,libspe,libauc,libf1score,libXROC,libYROC] = gridSearch(MRIK_tr,PETK_tr,MRIK_te,PETK_te,train_size,test_size,y_lasso_label,y_lasso_latest);
%                     [libacc,libsen,libspe] = gridSearchforSNP(MRIK_tr,PETK_tr,MRIK_te,PETK_te,SNPK_tr,SNPK_te,train_size,test_size,y_lasso_label,y_lasso_latest);
                    [libacc,libsen,libspe,libauc,libf1score,libXROC,libYROC,liby_true,liby_pred] = gridsearchfold(MRIK_tr,PETK_tr,MRIK_te,PETK_te,train_size,test_size,y_lasso_label,y_lasso_latest);
                    
                    Mdl = fitcsvm(X_train, y_lasso_label,'Standardize',true,'BoxConstraint',10);
                    [pred_label,score] = predict(Mdl,X_test);
                    TP = sum((pred_label ~= 1) & (y_lasso_latest ~= 1)); % caculate TP
                    TN = sum((pred_label == 1) & (y_lasso_latest == 1)); 
                    FP = sum((pred_label ~= 1) & (y_lasso_latest == 1)); 
                    FN = sum((pred_label == 1) & (y_lasso_latest ~= 1)); 
                    cross_validata_acc = cross_validata_acc + ((TP + TN) / (TP + TN + FP + FN));
                    cross_validata_sen = cross_validata_sen + (TP / (TP + FN));
                    cross_validata_spe = cross_validata_spe + (TN / (TN + FP));
    
                    libcross_validata_acc = libcross_validata_acc + libacc;
                    libcross_validata_sen = libcross_validata_sen + libsen;
                    libcross_validata_spe = libcross_validata_spe + libspe;
                    libcross_validata_auc = libcross_validata_auc + libauc;
                    libcross_validata_f1score = libcross_validata_f1score + libf1score;
%                     % Convert into 20x1 vector
                    paddedVectorX = [libXROC; zeros(35 - length(libXROC), 1)];
                    paddedVectorY = [libYROC; zeros(35 - length(libYROC), 1)];
                    XROC_ALL(:,c_v) = paddedVectorX;
                    YROC_ALL(:,c_v) = paddedVectorY;
                    %%% y_true & y_label also
                    paddedy_true = [liby_true;zeros(16 - length(liby_true),1)];
                    paddedy_pred = [liby_pred;zeros(16 - length(liby_pred),1)];
                    

                    each_cvacc(c_v) = ((TP + TN) / (TP + TN + FP + FN));
                    each_cvsen(c_v) = (TP / (TP + FN));
                    each_cvspe(c_v) = (TN / (TN + FP));
                    
                    MyMethody_trueall(:,c_v) = paddedy_true;
                    MyMethody_predall(:,c_v) = paddedy_pred;
                 
                    libeach_cvacc(c_v) = libacc;
                    libeach_cvsen(c_v) = libsen;
                    libeach_cvspe(c_v) = libspe;
                    libeach_cvauc(c_v) = libauc;
                    libeach_cvf1score(c_v) = libf1score;
               end
                
               
               if libcross_validata_auc / 10 > 0.80 & libcross_validata_auc / 10 < 0.81
                   
                    
                    MyMethody_trueall = zeros(35,10);
                    MyMethody_predall = zeros(35,10);
                    
                    
                     plot(XROC_ALL(:,1),YROC_ALL(:,1),'k-','linewidth',1.5);
                     hold on;
                     xlabel('FPR');ylabel('TPR');
                     legend('Mymethod');
                     tempauc = trapz(libXROC, libYROC);
                     XROC_ALL = zeros(35,10);
                     YROC_ALL = zeros(35,10);
               else
                    XROC_ALL = zeros(35,10);
                    YROC_ALL = zeros(35,10);
                    MyMethody_trueall = zeros(16,10);
                    MyMethody_predall = zeros(16,10);
                    
               end
               
               
               
                ACC(i,j,k,l) = cross_validata_acc / 10;
                SEN(i,j,k,l) = cross_validata_sen / 10;
                SPE(i,j,k,l) = cross_validata_spe / 10;
                
                libACC(i,j,k,l) = libcross_validata_acc / 10;
                libSEN(i,j,k,l) = libcross_validata_sen / 10;
                libSPE(i,j,k,l) = libcross_validata_spe / 10;
                libAUC(i,j,k,l) = libcross_validata_auc / 10;
                
                
%                 if libcross_validata_acc / 10 > 79
%                     disp('here');
%                 end
                if libcross_validata_auc / 10 > 80 && std(libeach_cvacc) < 9
                    disp('here');
                end
                
%                 if (libcross_validata_acc / 10) > 93
%                     save('NC_ADtheMRIROI.mat','MRIROI_array');
%                     save('NC_ADthePETROI.mat','PETROI_array');
%                     save('WForcheck.mat','W_t');
%                     save('WPETForcheck.mat','W_tpet');
%                 else
%                     MRIROI_array = zeros(30,10);
%                     PETROI_array = zeros(30,10);
%                 end
%                 

                % set zero
                libF1score(i,j,k,l) = libcross_validata_f1score / 10;
                libaccstd(i,j,k,l) = std(libeach_cvacc);
                libsenstd(i,j,k,l) = std(libeach_cvsen);
                libspestd(i,j,k,l) = std(libeach_cvspe);
                libaucstd(i,j,k,l) = std(libeach_cvauc);
                libf1scorestd(i,j,k,l) = std(libeach_cvf1score);
                accstd(i,j,k,l) = std(each_cvacc);
                senstd(i,j,k,l) = std(each_cvsen);
                spestd(i,j,k,l) = std(each_cvspe);
                cross_validata_acc = 0;
                cross_validata_sen = 0;
                cross_validata_spe = 0;
                libcross_validata_acc = 0;
                libcross_validata_sen = 0;
                libcross_validata_spe = 0;
                libcross_validata_auc = 0;
                libcross_validata_f1score = 0;
                MyMethody_trueall = [];
                MyMethody_predall=  [];
                libeach_cvacc = [];
                libeach_cvsen = [];
                libeach_cvspe = [];
                libeach_cvauc = [];
                libeach_cvf1score = [];
                each_cvacc = [];
                each_cvsen = [];
                each_cvspe = [];
            end
        end
    end
end








