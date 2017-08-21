clear
addpath(genpath('/home/wang/Documents/ECCV2016/camera_ready_version/sds_eccv2014-master'));
imgdir='/home/wang/Documents/ECCV2016/camera_ready_version/wzy/newimages/';
maskdir='/home/wang/Documents/ECCV2016/camera_ready_version/wzy/newimagesmaskinfo/';
featdir='./cnn/wzy/newimagesfeat/';


%%%%%%%%%%%%%
%Load network
%%%%%%%%%%%%%
model_def_file='prototxts/pinetwork_extract_fc7.prototxt';
model_file='sds_pretrained_models/nets/C';
assert(exist(model_def_file, 'file')>0);
assert(exist(model_file, 'file')>0);
rcnn_model=rcnn_create_model(model_def_file,model_file);
rcnn_model=rcnn_load_model(rcnn_model);
for idgb=1:50
    idgb
        idg=num2str(idgb);
        imnames=dir([imgdir idg '/' '*' '.bmp']);
        mkdir([featdir idg]);

%%%%%%%%%%%%%%%%%
%Extract features
%%%%%%%%%%%%%%%%%
for ii=1:length(imnames)
    disp(ii)
    imname=[imgdir idg '/' imnames(ii).name];
    img=imread(imname);
    
    %load mask
    maskname=[maskdir idg '/' imnames(ii).name(1:end-4) '_mask_info.mat'];
    load(maskname);
    sp=mask_info.sp;
    reg2sp=mask_info.reg2sp;
    boxes=get_region_boxes(sp, reg2sp);
    
    %pass it through rcnn
    rcnnfeats=rcnn_features_pi(img, sp, reg2sp, boxes, rcnn_model);

    %save features
%     save(fullfile(featdir, [imnames(ii).name(1:end-4) '.mat']), 'rcnnfeats');
  save([featdir idg '/' imnames(ii).name(1:end-4) '.mat'], 'rcnnfeats');
    clear rcnnfeats
    
end 
end