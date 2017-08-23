
clc;clear;close all;
addpath(genpath('./sds_eccv2014-master'));
addpath(genpath('./gop_1.3'));


train_path='./data/MSRA-5000/';    % the training images file path
imagt_path='./data/MSRA-5000-gt/'; % the ground truth file path
% sp_path='./code_superpixels/sp/MSRA/';  % the superpixels file path
train_listname = textread('./train_list.txt','%s'); % the names list of training data
mkdir('./trained_model');
co_sum=0;ro_sum=0;sam_pri=[];sam_inf=[];
ini_ro=1;
ros=1;cols=1;

%% Load R-CNN network
model_def_file='./sds_eccv2014-master/prototxts/pinetwork_extract_fc7.prototxt';
model_file='./sds_eccv2014-master/sds_pretrained_models/nets/C';
assert(exist(model_def_file, 'file')>0);
assert(exist(model_file, 'file')>0);
caffe.set_mode_gpu();
caffe.set_device(0);
rcnn_model=rcnn_create_model(model_def_file,model_file);
rcnn_model=rcnn_load_model(rcnn_model);


for it=1:length(train_listname)
    it
    imgname=train_listname{it};
    img_name=[train_path imgname(1:end-3) 'jpg'];
    imggt_name=[imagt_path imgname(1:end-3) 'png'];
%     sp_name=[sp_path imgname(1:end-4) '_sp.mat'];
    I=imread(img_name);
    
    %% extract proposals and preprocess proposals
    [masks]=extract_proposal(I);
    
    %% generate positive and negative samples
    I_gt=im2double(imread(imggt_name));
    I_gt=I_gt(:,:,1);
    num_gtpixel=length(find(I_gt==1));   
    pros_num=size(masks,3);
    clear mask
    clear O_rank
    clear boxes
    %% compute confidence scores
    for j=1:pros_num       
        pro=masks(:,:,j);
        s=I_gt+pro;
        num_salpixel=length(find(s==2));
        num_propixel=length(find(pro==1));
        beta=0.3;
        conf(j)=(1+beta)*num_salpixel/(beta*num_gtpixel+num_propixel);
    end
    [conf,ind]=sort(conf,'descend');
    masks=masks(:,:,ind);
    
    %% positive samples: confidence scores>=0.9
    %% negative samples: confidence scores<=0.6
    pos_ind=find(conf>=0.9);
    pos_num=length(pos_ind);
    if pos_num<2   
        continue
    else
        
        %% randomly select negative samples between confidence scores <=0.6
        %% the number of positive samples equals the number of negative samples
        neg_ind=find(conf<=0.6);
        neg_num=length(neg_ind);
        sam=randperm(neg_num);
        sam_ind=sort(sam(1:pos_num));
        cc=size(conf,2);
        conf=[conf(1:pos_num) conf(cc-neg_num+sam_ind)];
        mask(:,:,1:pos_num)=masks(:,:,1:pos_num);
        mask(:,:,pos_num+1:pos_num*2)=masks(:,:,cc-neg_num+sam_ind);
        neg_ind=find(conf<=0.6);
        neg_num=length(neg_ind);
        
        %% construct the matrix of preference pairs, see details
        %% in 'ranksvm.m'
        k=1;
        for j=1:pos_num
            for w=1:neg_num              
                O_rank(k,pos_ind(j))=1;
                O_rank(k,neg_ind(w))=-1;
                k=k+1;
            end
        end


        %% extract superpixels
%         [segimage,spnum] = slicmex(I,400,20);
%         segimage=double(segimage);
%         segimage=segimage+1;
%         nseg=max(segimage(:));
%         for k=1:nseg
%             nu_pixel(1,k)=length(find(segimage(:)==k));
%         end
%         sp=load(sp_name);
%         seg=sp.imsegs;
        pro_num=size(mask,3);
%         mask_info.sp=seg.segimage;
%         mask_info.sp=segimage;
%         mask_info.reg2sp=zeros(nseg,pro_num);
%         mask_info.reg2sp=zeros(seg.nseg,pro_num);  
        
        %% prepare data for feature extraction
%         for j=1:pro_num
%             I_pro=mask(:,:,j);
% %             pro_sp=seg.segimage.*I_pro;
%             pro_sp=segimage.*I_pro;
%             bb= tabulate(pro_sp(:));
% %             [c,ia,ib]=intersect(bb(:,1),[1:seg.nseg]);
%             [c,ia,ib]=intersect(bb(:,1),[1:nseg]);
%             tt=bb(:,2)';
%             ratio=tt(ia)./nu_pixel(ib);
%             pro_sp_ind=find(ratio>0.6);
%             mask_info.reg2sp(ib(pro_sp_ind),j)=1;
%         end
        
        %% extract R-CNN features
%         sp=mask_info.sp;
%         reg2sp=mask_info.reg2sp;
%         boxes=get_region_boxes(sp, reg2sp);
        for j=1:pro_num
            [ross,colss]=find(mask(:,:,j)==1);

             boxes(j,:)=[min(colss(:)),min(ross(:)),max(colss(:)),max(ross(:))];
        end
%         feat=rcnn_features_pi(I, sp, reg2sp, boxes, rcnn_model);
        mask=double(mask);
        feat=rcnn_features_pi_2(I, mask, boxes, rcnn_model);   
        
        pro_num=size(feat,1);
        sam_pri=[sam_pri;feat(1:pro_num/2,:)];
        sam_inf=[sam_inf;feat(pro_num/2+1:end,:)];
        [ro,co]=size(O_rank); 
        co_sum=co_sum+co/2;
        ro_sum=ro_sum+ro;
        po(ros:ro_sum,cols:co_sum)=O_rank(:,1:co/2);
        ne(ros:ro_sum,cols:co_sum)=O_rank(:,co/2+1:end);
        for k=cols:co_sum
            for w=k+1:co_sum
                ix_pair_po_pri(ini_ro,1)=k;
                ix_pair_po_pri(ini_ro,2)=w;
                ini_ro=ini_ro+1;             
            end
        end
        ros=ros+ro;
        cols=cols+co/2;
       
          
    end

end
caffe.reset_all();
%% set parameters 
option.beta=3;
option.d=120;
option.epsilon=1.0000e-4;
option.lambda=0.01;
option.kernel='gaus-rbf';

O_pair=sparse(ro_sum,2*co_sum);
ix_pair_po_inf=ix_pair_po_pri+co_sum;
O_pair=[po ne];
co_sum=size(O_pair,2);

%% construct positive sample pairs
ix_pair_po=[ix_pair_po_pri;ix_pair_po_inf];


%% construct negative sample pairs
[r1,c1]=(find(O_pair==1));
for i=1:size(O_pair,1)
    [r2, c2(i,1)]=(find(O_pair(i,:)==-1));
end
ix_pair_ne=[c1 c2];

%% construct sample pairs
ix_pair=[ix_pair_po;ix_pair_ne];

%% construct label matrix
y1=ones(size(ix_pair_po,1),1);
y2=-1*ones(size(ix_pair_ne,1),1);
y=[y1;y2];

C_O=0.0001*ones(size(O_pair,1),1);

%% pca dimension reduction
sam_train=[sam_pri;sam_inf];
[coeff,score,latent] = pca(sam_train);
lat = cumsum(latent)./sum(latent);
dim = find(lat>0.85,1);
parts_train = score(:,1:dim)';
train=double(parts_train');
save('./trained_model/coeff.mat','coeff');

%% compute the kernel matrix
Method = struct('rbf_sigma',0);
[K, Method] = ComputeKernel(train, option.kernel, Method);
K= K*size(K,1)/trace(K); % scale the kernel matrix

%% K*J_n in PCCA paper equation(10), see reference [52]
% save('./K.mat','K');
KJ= sparse([], [], [], size(ix_pair, 1), size(train,1)^2, 0);
chop_num = 2000;  % chop into samll piece for speed up (2000 needs 17GB for CAVIAR)
for ns = 1:chop_num:size(ix_pair,1)%-mod(size(ix_pair,1),chop_num)
    chop_mat = sparse(chop_num,size(train,1)^2);
    n = 1;
    for i = ns:min(chop_num+ns-1,size(ix_pair,1))
        %         chop_row = zeros(1,size(X,1)^2);
        ix_row1= sub2ind(size(K), 1:size(train,1), ones(1,size(train,1))*ix_pair(i,1));
        %         chop_row(1,ix_row1) = K(:,ix_pair(ns+i-1,1))- K(:,ix_pair(ns+i-1,2));
        chop_mat(n,ix_row1) = K(:,ix_pair(i,1))- K(:,ix_pair(i,2));
        ix_row2= sub2ind(size(K), 1:size(train,1), ones(1,size(train,1))*ix_pair(i,2));
        %         chop_row(1,ix_row2) = -chop_row(1,ix_row1);
        chop_mat(n,ix_row2) = -chop_mat(n,ix_row1);
        %         chop_mat(i,:) = chop_row;        
        n = n+1;
    end
    if chop_num+ns-1 < size(ix_pair,1)
        KJ(ns:chop_num+ns-1,:) = sparse(chop_mat);
    else 
        KJ(ns:size(ix_pair,1),:) = sparse(chop_mat(1:mod(size(ix_pair,1),chop_num),:));
    end
end
% save('./KJ.mat','KJ');

%% initialize Method
[Method]=PCCA(train,ix_pair,y,option,K,Method,KJ);

C=max(C_O(:));
iter=0;
l_old=0;
w = zeros(option.d,1);
A=Method.P;

[K_test] = compute_rank2_new(Method, train, train);
% save('./K_new.mat','K_test');

%% set the initial metric loss to 0
Method.l_met=0;

%% joint learning
while(1) 
    iter=iter+1

    X_new=A*K_test;
    X_new=double(X_new);
    
    %% fix P, update w, see equantio (8)
    w = ranksvm(X_new',O_pair,C_O,w,Method.l_met);
    
    %% fix w, update P, see equation (10)
    [Method]=PCCA_new(train,ix_pair,y,option,w,ix_pair_ne,C,Method,K,KJ);
    
    %% iteration stopping criterion
    if abs(Method.l_new-l_old)<option.epsilon && norm(A-Method.P, 'fro')/norm(A, 'fro')<option.epsilon
        break;
    end

    A=Method.P;
    l_old=Method.l_new;
end

%% save train data
save('./trained_model/w.mat','w');
save('./trained_model/train.mat','train');
save('./trained_model/Method.mat','Method');
caffe.reset_all();
