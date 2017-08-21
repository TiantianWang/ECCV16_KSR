
%% test
clc;clear;close all;
addpath(genpath('./sds_eccv2014-master'));
addpath(genpath('./gop_1.3'));

test_path='./data/SOD_bmp/';
% sp_path='./code_superpixels/sp/SOD/';
fusion_path='./saliency_map/SOD/';
mkdir(fusion_path);
test_dir=dir(test_path);
test_num=length(test_dir);

%% Load network
model_def_file='./sds_eccv2014-master/prototxts/pinetwork_extract_fc7.prototxt';
model_file='./sds_eccv2014-master/sds_pretrained_models/nets/C';
assert(exist(model_def_file, 'file')>0);
assert(exist(model_file, 'file')>0);
caffe.set_mode_gpu();
caffe.set_device(0);
rcnn_model=rcnn_create_model(model_def_file,model_file);
rcnn_model=rcnn_load_model(rcnn_model);


%% load trained model
load('./trained_model/coeff.mat');
load('./trained_model/Method.mat');
load('./trained_model/train.mat');
load('./trained_model/w.mat');
dim=size(train,2);
tic();
for it=1:test_num
    it
    imgname=test_dir(it+2).name;
    img_name=[test_path imgname];
%     sp_name=[sp_path imgname(1:end-4) '_sp.mat'];  
    fusion_name=[fusion_path imgname(1:end-4) '.png'];
    I=imread(img_name); 
   if ~exist(fusion_name)
        
        %% extract proposals and preprocess proposals
        [masks]=extract_proposal(I);
% 
%                %% extract superpixels
%         [segimage,spnum] = slicmex(I,400,20);
%         segimage=double(segimage);
%         segimage=segimage+1;
%         nseg=max(segimage(:));
%         for k=1:nseg
%             nu_pixel(1,k)=length(find(segimage(:)==k));
%         end
%         sp=load(sp_name);
%         seg=sp.imsegs;
        pros_num=size(masks,3);
%         mask_info.sp=seg.segimage;
%         mask_info.sp=segimage;
%         mask_info.reg2sp=zeros(nseg,pros_num);
%         mask_info.reg2sp=zeros(seg.nseg,pro_num);  
        
        %% prepare data for feature extraction
%         md=size(masks,3);
        for j=1:pros_num
            [ros,cols]=find(masks(:,:,j)==1);

             boxes(j,:)=[min(cols(:)),min(ros(:)),max(cols(:)),max(ros(:))];
        end
% % %         for j=1:pros_num
% % %             I_pro=masks(:,:,j);
% % % %             pro_sp=seg.segimage.*I_pro;
% % %             pro_sp=segimage.*I_pro;
% % %             bb= tabulate(pro_sp(:));
% % % %             [c,ia,ib]=intersect(bb(:,1),[1:seg.nseg]);
% % %             [c,ia,ib]=intersect(bb(:,1),[1:nseg]);
% % %             tt=bb(:,2)';
% % %             ratio=tt(ia)./nu_pixel(ib);
% % %             pro_sp_ind=find(ratio>0.6);
% % %             mask_info.reg2sp(ib(pro_sp_ind),j)=1;
% % %         end
% % %         
% % %         %% extract R-CNN features
% % %         sp=mask_info.sp;
% % %         reg2sp=mask_info.reg2sp;
% % %         boxes=get_region_boxes(sp, reg2sp);
        masks=double(masks);
        feats=rcnn_features_pi_2(I, masks, boxes, rcnn_model);

        %% pca dimension reduction
        feats=bsxfun(@minus,feats,mean(feats))*coeff(:,1:dim);
        sam_test=double(feats);
        
        %% computer the kernel matrix
        [Test_new] = compute_rank2(Method, train, sam_test);
      
        %% compute ranking scores
        rank_val=Test_new'*w;

        %% final saliency map: a weighted fusion of top-16 ranked masks;
        mask_num=size(masks,3);
        [rank_val,inde]=sort(rank_val,'descend');
        masks=masks(:,:,inde);
        rank=rank_val(1:16);
        [rows,cols]=size(masks(:,:,1));
        result=zeros(rows,cols);
        for j=1:16
            result=result+masks(:,:,j)*exp(2*rank(j));
        end
        
        %% normalization
        index=find(result~=0);
        mi=result(index);
        result=(result-min(mi(:)))/(max(mi(:))-min(mi(:)));
        ind=find(result==(-min(mi(:)))/(max(mi(:))-min(mi(:))));
        result(ind)=0;
     
        imwrite(result,fusion_name,'png');
        clear boxes
        
   end 
    
end
toc();
