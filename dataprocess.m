load('./ADNIFullDataset.mat');
Y_PET_baseline = ADNIFullData.PET.data;
X_MRI_baseline = ADNIFullData.PET.data(1);
X_MRI_baseline = ADNIFullData.MRI.data(1);
X_MRI_06 = ADNIFullData.MRI.data(2);
X_MRI_12 = ADNIFullData.MRI.data(3);
label = ADNIFullData.label;
load('./My_ADNI_805_ALL.mat');
age = My_ADNI_805_ALL(:,578);
gender = My_ADNI_805_ALL(:,577);



% 
% 
% % 删除缺失的PET样本数据和对应的MRI样本数据
y_pet_matrix = cell2mat(Y_PET_baseline);
y_pet_baseline = cell2mat(Y_PET_baseline);
load('./SNPs.mat');
SNP_data = OriginalSNPs100;
[n,m] = size(y_pet_matrix);
% 
X_mri_baseline = cell2mat(X_MRI_baseline);
% y_pet_baseline = cell2mat(Y_PET_baseline);
% 
d = all(~isnan(y_pet_baseline),2);
snp_d = all(~isnan(SNP_data),2);
y_pet_del = y_pet_baseline(d,:);
X_mri_pet = X_mri_baseline(d,:);
X_mri_pet_snp = X_mri_baseline(d & snp_d,:);
y_pet_snp = y_pet_baseline(d & snp_d,:);
SNP_mri_pet = SNP_data(d & snp_d,:);
label_snp = label(d & snp_d);
age_d = age(d);
gender_d = gender(d);

% 
% 
% % 删除缺失的CSF数据和对应的MRI样本数据
% CSF_matrix = cell2mat(CSF_baseline);
% [n,m] = size(CSF_matrix);
% 
% X_mri_baseline2 = cell2mat(X_MRI_baseline);
% 
% 
% e = all(~isnan(CSF_matrix),2);
% CSF_matrix = CSF_matrix(e,:);
% X_mri_baseline2 = X_mri_baseline2(e,:);
% 
% mri_06 = cell2mat(X_MRI_06);
% mri_12 = cell2mat(X_MRI_12);
% c = all(~isnan(mri_06),2);
% e = all(~isnan(mri_12),2);
% new_del = c & e & d;
% X_mri_time_baseline = X_mri_baseline(new_del,:);
% mri_06 = mri_06(new_del,:);
% mri_12 = mri_12(new_del,:);
% y_pet_tmie = y_pet_baseline(new_del,:);

clinical_score = ADNIFullData.clinicalScores.data(1);
clinical_score = cell2mat(clinical_score);
clinical_pet_mri = clinical_score(d,:);

labeldpet = label(d);
label_snp_pet = label(d & snp_d);
clinical_snp = clinical_score(d & snp_d,:);

MCI_NC_note = find(label < 2);
MCI_NC_pet_note = find(labeldpet < 2);
MCI_NC_label = label(MCI_NC_note);
MCI_NC_mri = X_mri_baseline(MCI_NC_note,:);
MCI_NC_clinical = clinical_score(MCI_NC_note,:);
MCI_NC_mri_pet = X_mri_pet(MCI_NC_pet_note,:);
MCI_NC_pet = y_pet_del(MCI_NC_pet_note,:);
MCI_NC_dPET_label = labeldpet(MCI_NC_pet_note);
MCI_NC_clinical_pet = clinical_pet_mri(find(labeldpet<2),:);
MCI_NC_andPET_age = age(MCI_NC_pet_note);
MCI_NC_andPET_gender = gender(MCI_NC_pet_note);



MCI_NC_mri_pet_snp = X_mri_pet_snp(find(label_snp_pet < 2),:);
MCI_NC_pet_snp = y_pet_snp(find(label_snp_pet < 2),:);
MCI_NC_snp = SNP_mri_pet(find(label_snp_pet < 2),:);
MCI_NC_snp_label = label_snp_pet(find(label_snp_pet < 2));



MCI_AD_note = find(labeldpet > 0);
MCI_AD_mri_base =  X_mri_baseline(find(label>0),:);
MCI_AD_clinical = clinical_score(find(label>0),:);
MCI_AD_label = label(find(label>0));
MCI_AD_dPET_label = labeldpet(MCI_AD_note);
MCI_AD_mri = X_mri_pet(MCI_AD_note,:);
MCI_AD_pet = y_pet_del(MCI_AD_note,:);
MCI_AD_clinical_pet = clinical_pet_mri(MCI_AD_note,:);
MCI_AD_andPET_age = age(MCI_AD_note);
MCI_AD_andPET_gender = gender(MCI_AD_note);


% 

% X_mri_snp = X_mri_baseline(snp_d,:);
% SNP_data = SNP_data(snp_d,:);
% labelsnp = label(snp_d);
% MCI_AD_note = find(labelsnp > 0);
% MCI_AD_label = labelsnp(MCI_AD_note);
% MCI_AD_mri = X_mri_snp(MCI_AD_note,:);
% MCI_AD_snp = SNP_data(MCI_AD_note,:);



NC_AD_note = find(labeldpet ~= 1);
NC_AD_pet_label = labeldpet(NC_AD_note);
NC_AD_mri_pet = X_mri_pet(NC_AD_note,:);
NC_AD_pet = y_pet_del(NC_AD_note,:);
NC_AD_pet_mri_clinical = clinical_pet_mri(NC_AD_note,:);
NC_AD_clinical = clinical_score(find(label ~= 1),:);
NC_AD_mri = X_mri_baseline(find(label ~= 1),:);
NC_AD_label = label(find(label ~= 1));
NC_AD_andPET_age = age(NC_AD_note);
NC_AD_andPET_gender = gender(NC_AD_note);
NC_age = NC_AD_andPET_age(~logical(NC_AD_pet_label));
AD_age = NC_AD_andPET_age(logical(NC_AD_pet_label)); 
NC_gender = NC_AD_andPET_gender(~logical(NC_AD_pet_label));
AD_gender = NC_AD_andPET_gender(logical(NC_AD_pet_label));
NC_MMSE = NC_AD_pet_mri_clinical(~logical(NC_AD_pet_label));
AD_MMSE = NC_AD_pet_mri_clinical(logical(NC_AD_pet_label));


NC_AD_mri_pet_snp = X_mri_pet_snp(find(label_snp_pet ~= 1),:);
NC_AD_pet_snp = y_pet_snp(find(label_snp_pet ~= 1),:);
NC_AD_snp = SNP_mri_pet(find(label_snp_pet ~= 1),:);
NC_AD_label_snp = label_snp_pet(find(label_snp_pet ~= 1));


% MRI with PET and SNP data MCI and AD
mri_pet_snp_d = d & snp_d;
Y_msplabel = label(mri_pet_snp_d);
MCI_AD_note_MPS = find(Y_msplabel > 0);
X_mrips = X_mri_baseline(mri_pet_snp_d,:);
X_petms = y_pet_baseline(mri_pet_snp_d,:);
X_snpmp = SNP_data(mri_pet_snp_d,:);
X_mpsclinical = clinical_score(mri_pet_snp_d,:);

MCI_AD_mrips = X_mrips(MCI_AD_note_MPS,:);
MCI_AD_petms = X_petms(MCI_AD_note_MPS,:);
MCI_AD_snpmp = X_snpmp(MCI_AD_note_MPS,:);
MCI_AD_mpsclinical = X_mpsclinical(MCI_AD_note_MPS,:);
MCI_AD_mpslabel = Y_msplabel(MCI_AD_note_MPS);

% 
% sMCI 和 pMCI
logi_pMCI = logical(ADNIFullData.pMCI);

MCI_data = label;
MCI_data(MCI_data ~= 1) =0;
sMCI_data = xor(MCI_data,logi_pMCI); %% 通过异或运算来求sMCI
sMCI_MRIdata = X_mri_baseline(sMCI_data);
MCI_label = label(logical(MCI_data));
pMCI_sMCI_label = MCI_label & ADNIFullData.pMCI(logical(MCI_data));
pMCI_sMCI_label = double(pMCI_sMCI_label);
pMCI_sMCI_MRIdata = X_mri_baseline(logical(MCI_data),:);


pMCI_sMCI_andPETlabel = labeldpet(find(labeldpet == 1));
pMCI_andPET = ADNIFullData.pMCI(d);
pMCI_andPET = pMCI_andPET(find(labeldpet == 1));
pMCI_sMCI_andPET_label = pMCI_sMCI_andPETlabel & pMCI_andPET;
pMCI_sMCI_andPET_label = double(pMCI_sMCI_andPET_label);
pMCI_sMCI_andPETMRIdata = X_mri_pet(find(labeldpet == 1),:);
pMCI_sMCI_PETdata = y_pet_del(find(labeldpet == 1),:);
pMCI_sMCI_clinicaldata = clinical_pet_mri(find(labeldpet == 1),:);
pMCI_sMCI_andPET_age = age(d);
pMCI_sMCI_andPET_age = pMCI_sMCI_andPET_age(find(labeldpet == 1));
pMCI_sMCI_andPET_gender = gender(d);
pMCI_sMCI_andPET_gender = pMCI_sMCI_andPET_gender(find(labeldpet == 1));

pMCI_age = pMCI_sMCI_andPET_age(logical(pMCI_sMCI_andPET_label));
sMCI_age = pMCI_sMCI_andPET_age(~logical(pMCI_sMCI_andPET_label));
pMCI_gender = pMCI_sMCI_andPET_gender(logical(pMCI_sMCI_andPET_label));
sMCI_gender = pMCI_sMCI_andPET_gender(~logical(pMCI_sMCI_andPET_label));
pMCI_MMSE = pMCI_sMCI_clinicaldata(logical(pMCI_sMCI_andPET_label));
sMCI_MMSE = pMCI_sMCI_clinicaldata(~logical(pMCI_sMCI_andPET_label));


pMCI_sMCI_andPETSNPlabel = labeldpet(find(label_snp_pet == 1));
pMCI_andPETSNP = ADNIFullData.pMCI(d & snp_d);
pMCI_andPETSNP = pMCI_andPETSNP(find(label_snp_pet == 1));
pMCI_sMCI_andPETSNP_label = pMCI_sMCI_andPETSNPlabel & pMCI_andPETSNP;
pMCI_sMCI_andPETSNP_label = double(pMCI_sMCI_andPETSNP_label);
pMCI_sMCI_andPETMRISNPdata = X_mri_pet_snp(find(label_snp_pet == 1),:);
pMCI_sMCI_PETSNPdata = y_pet_snp(find(label_snp_pet == 1),:);
pMCI_sMCI_SNPdata = SNP_mri_pet(find(label_snp_pet == 1),:);
pMCI_sMCI_SNPclinical = clinical_snp(find(label_snp_pet == 1),:);

pMCI_sMCI_ALL = [pMCI_sMCI_andPETMRIdata pMCI_sMCI_PETdata pMCI_sMCI_clinicaldata pMCI_sMCI_andPET_label];
onlypMCI_MRI_PET_CLI = pMCI_sMCI_ALL(pMCI_sMCI_ALL(:, end) == 1, :);
onlysMCI_MRI_PET_CLI = pMCI_sMCI_ALL(pMCI_sMCI_ALL(:, end) == 0, :);
onlyNC_ALL = [X_mri_pet y_pet_del clinical_pet_mri labeldpet];
onlyNC_MRI_PET_CLI = onlyNC_ALL(onlyNC_ALL(:,end)== 0,:);

NC_pMCI_MRI = [onlypMCI_MRI_PET_CLI(:,1:93);onlyNC_MRI_PET_CLI(:,1:93)];
NC_pMCI_PET = [onlypMCI_MRI_PET_CLI(:,94:186);onlyNC_MRI_PET_CLI(:,94:186)];
NC_pMCI_label = [ones(size(onlypMCI_MRI_PET_CLI,1),1);zeros(size(onlyNC_MRI_PET_CLI,1),1)];
NC_pMCI_clinical = [onlypMCI_MRI_PET_CLI(:,187:191);onlyNC_MRI_PET_CLI(:,187:191)];

NC_sMCI_MRI = [onlysMCI_MRI_PET_CLI(:,1:93);onlyNC_MRI_PET_CLI(:,1:93)];
NC_sMCI_PET = [onlysMCI_MRI_PET_CLI(:,94:186);onlyNC_MRI_PET_CLI(:,94:186)];
NC_sMCI_label = [ones(size(onlysMCI_MRI_PET_CLI,1),1);zeros(size(onlyNC_MRI_PET_CLI,1),1)];
NC_sMCI_clinical = [onlysMCI_MRI_PET_CLI(:,187:191);onlyNC_MRI_PET_CLI(:,187:191)];

pMCI_sMCI_SNPALL = [pMCI_sMCI_andPETMRISNPdata pMCI_sMCI_PETSNPdata pMCI_sMCI_SNPdata pMCI_sMCI_SNPclinical pMCI_sMCI_andPETSNP_label];
onlyNC_SNPALL = [X_mri_pet_snp y_pet_snp SNP_mri_pet clinical_snp label_snp_pet];
onlysMCI_MRI_PET_SNP = pMCI_sMCI_SNPALL(pMCI_sMCI_SNPALL(:, end) == 0, :);
onlyNC_MRI_PET_SNP = onlyNC_SNPALL(onlyNC_SNPALL(:,end)== 0,:);
NC_sMCI_SNPMRI = [onlysMCI_MRI_PET_SNP(:,1:93);onlyNC_MRI_PET_SNP(:,1:93)];
NC_sMCI_SNPPET = [onlysMCI_MRI_PET_SNP(:,94:186);onlyNC_MRI_PET_SNP(:,94:186)];
NC_sMCI_SNPlabel = [ones(size(onlysMCI_MRI_PET_SNP,1),1);zeros(size(onlyNC_MRI_PET_SNP,1),1)];
NC_sMCI_SNPclinical = [onlysMCI_MRI_PET_SNP(:,287:291);onlyNC_MRI_PET_SNP(:,287:291)];
NC_sMCI_SNPSNP = [onlysMCI_MRI_PET_SNP(:,187:286);onlyNC_MRI_PET_SNP(:,187:286)];


onlyAD_ALL = [X_mri_pet y_pet_del labeldpet];
onlyAD_MRI_PET = onlyAD_ALL(onlyAD_ALL(:,end)== 2,:);
% 不许动，谁动谁是狗
sMCI_AD_MRI = [onlysMCI_MRI_PET_CLI(:,1:93);onlyAD_MRI_PET(:,1:93)];
sMCI_AD_PET = [onlysMCI_MRI_PET_CLI(:,94:186);onlyAD_MRI_PET(:,94:186)];
sMCI_AD_label = [zeros(size(onlysMCI_MRI_PET_CLI,1),1);ones(size(onlyAD_MRI_PET,1),1)];

pMCI_AD_MRI = [onlypMCI_MRI_PET_CLI(:,1:93);onlyAD_MRI_PET(:,1:93)];
pMCI_AD_PET = [onlypMCI_MRI_PET_CLI(:,94:186);onlyAD_MRI_PET(:,94:186)];
pMCI_AD_label = [zeros(size(onlypMCI_MRI_PET_CLI,1),1);ones(size(onlyAD_MRI_PET,1),1)];








    
        