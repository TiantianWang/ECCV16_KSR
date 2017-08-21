clear
addpath(genpath('/home/wang/Desktop/camera_ready_version/sds_eccv2014-master'));
imgdir='/home/wang/Desktop/camera_ready_version/data/MSRA-5000/';
maskdir='/home/wang/Desktop/MSRA-5000_0.9/';
featdir='/home/wang/Desktop/camera_ready_version/cnn/pro_sp/MSRA-5000_0.9/';
mkdir(featdir)
% imnames=dir([imgdir '*' '.jpg']);
masknames=dir([maskdir '*' '.mat']);

% if exist('../+caffe', 'dir')
%   addpath('..');
% else
%   error('Please run this demo from caffe/matlab/demo');
% end
caffe.set_mode_gpu();
caffe.set_device(0);
%%%%%%%%%%%%%
%Load network
%%%%%%%%%%%%%
model_def_file='prototxts/pinetwork_extract_fc7.prototxt';
model_file='sds_pretrained_models/nets/C';
assert(exist(model_def_file, 'file')>0);
assert(exist(model_file, 'file')>0);
rcnn_model=rcnn_create_model(model_def_file,model_file);
rcnn_model=rcnn_load_model(rcnn_model);

%%%%%%%%%%%%%%%%%
%Extract features
%%%%%%%%%%%%%%%%%
for ii=1:2000
    disp(ii)
    imname=[imgdir masknames(ii).name(1:end-14) '.bmp'];
    img=imread(imname);
    
    %load mask
    maskname=[maskdir masknames(ii).name(1:end-14) '_mask_info.mat'];
    load(maskname);
    sp=mask_info.sp;
    reg2sp=mask_info.reg2sp;
    boxes=get_region_boxes(sp, reg2sp);
    
    %pass it through rcnn
    rcnnfeats=rcnn_features_pi(img, sp, reg2sp, boxes, rcnn_model);

    %save features
    save(fullfile(featdir, [masknames(ii).name(1:end-14) '.mat']), 'rcnnfeats');
    clear rcnnfeats
    
end   