function [window, masked_window] = rcnn_im_crop_mask(im, mask, bbox, crop_mode, crop_size, padding, image_mean)
%modified from rcnn
use_square = false;
if strcmp(crop_mode, 'square')
  use_square = true;
end

% defaults if padding is 0
pad_w = 0;
pad_h = 0;
crop_width = crop_size;
crop_height = crop_size;
if padding > 0 || use_square
  %figure(1); showboxesc(im/256, bbox, 'b', '-');
  scale = crop_size/(crop_size - padding*2);
  half_height = (bbox(4)-bbox(2)+1)/2;
  half_width = (bbox(3)-bbox(1)+1)/2;
  center = [bbox(1)+half_width bbox(2)+half_height];
  if use_square
    % make the box a tight square
    if half_height > half_width
      half_width = half_height;
    else
      half_height = half_width;
    end
  end
  bbox = round([center center] + ...
      [-half_width -half_height half_width half_height]*scale);
  unclipped_height = bbox(4)-bbox(2)+1;
  unclipped_width = bbox(3)-bbox(1)+1;

  %figure(1); showboxesc([], bbox, 'r', '-');
  pad_x1 = max(0, 1 - bbox(1));
  pad_y1 = max(0, 1 - bbox(2));
  % clipped bbox
  bbox(1) = max(1, bbox(1));
  bbox(2) = max(1, bbox(2));
  bbox(3) = min(size(im,2), bbox(3));
  bbox(4) = min(size(im,1), bbox(4));
  clipped_height = bbox(4)-bbox(2)+1;
  clipped_width = bbox(3)-bbox(1)+1;
    scale_x = crop_size/unclipped_width;
  scale_y = crop_size/unclipped_height;
  crop_width = round(clipped_width*scale_x);
  crop_height = round(clipped_height*scale_y);
  pad_x1 = round(pad_x1*scale_x);
  pad_y1 = round(pad_y1*scale_y);

  pad_h = pad_y1;
  pad_w = pad_x1;
  % TODO: handle flipping

  if pad_y1 + crop_height > crop_size
    crop_height = crop_size - pad_y1;
  end
  if pad_x1 + crop_width > crop_size
    crop_width = crop_size - pad_x1;
  end
end % padding > 0 || square

window = im(bbox(2):bbox(4), bbox(1):bbox(3), :);
mask_window = mask(bbox(2):bbox(4), bbox(1):bbox(3),:);


tmp = imresize(window, [crop_height crop_width], ...
    'bilinear', 'antialiasing', false);
if ~isempty(image_mean)
  tmp = tmp - image_mean(pad_h+(1:crop_height), pad_w+(1:crop_width), :);
end
mask_tmp=imresize(mask_window, [crop_height crop_width], 'nearest');
masked_tmp=tmp;
for ch=1:3, masked_tmp(:,:,ch)=tmp(:,:,ch).*mask_tmp; end

%figure(2); window_ = tmp; imagesc((window_-min(window_(:)))/(max(window_(:))-min(window_(:)))); axis image;
window = zeros(crop_size, crop_size, 3, 'single');
% window = zeros(crop_size, crop_size, 3, 'uint8');
window(pad_h+(1:crop_height), pad_w+(1:crop_width), :) = tmp;
%figure(3); imagesc((window-min(window(:)))/(max(window(:))-min(window(:)))); axis image; pause;

masked_window = zeros(crop_size, crop_size, 3, 'single');
% masked_window = zeros(crop_size, crop_size, 3, 'uint8');
masked_window(pad_h+(1:crop_height), pad_w+(1:crop_width), :) = masked_tmp;



