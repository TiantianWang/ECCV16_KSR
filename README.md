# Kernelized Subspace Ranking for Saliency Detection
This package has the source code for the paper "Kernelized Subspace Ranking for Saliency Detection" (ECCV16).

## Citing this work
If you find this work useful in your research, please consider citing:

     @inproceedings{wangeccv16,
        Author={Tiantian Wang and Lihe Zhang and Huchuan Lu and Chong Sun and Jinqing Qi},
        Title={Kernelized Subspace Ranking for Saliency Detection},
        Booktitle={European Conference on Computer Vision (ECCV)},
        Year={2016}
     }

## Installation
1. Install prerequsites for `Caffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
2. Compile the `./sds_eccv2014-master/extern/caffe-master`submodule.
3. Compile the `./gop_1.3` submodule.

## Prerequisites
Download pretrained SDS model from [Baidu Drive](https://pan.baidu.com/s/1ZIhSyF47PA-MwF77oKaqhw) or [Google Drive](https://pan.baidu.com/s/1ZIhSyF47PA-MwF77oKaqhw). Then put it into the `./sds_eccv2014-master` folder.

## Train & Test

1. Train: run `runme_train.m` to generate trained model in the `./trained_model` folder.
2. Test: run `runme_test.m` to generate saliency maps in the `./saliency_map` folder. 

## Our Trained Model
Download trained models from [Baidu Drive](http://pan.baidu.com/s/1boKHG2V) or [Google Drive](https://drive.google.com/open?id=0B_MpGgTntG47WVU3NEUwNlBYczg). Then put it into the `./trained_model` folder.

## Contact
tiantianwang.ice@gmail.com



