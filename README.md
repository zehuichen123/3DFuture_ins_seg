# 1st Place Solutions of 3D AI Challenge 2020 - Instance Segmentation Track

This repository maintains our solution to [3D AI Challenge Instance Segmentation Track](), which ranks 1st in both trackA (validation set)] and trackB (test set). 
Our solution is a weighted ensemble of several [PointRend]() models. This repository implements PointRend and train/evaluates it on the [3DFuture dataset](), 
based on the open projects [mmdetection]() and [detectron2 PointRend](). We also provide ensemble code on our trained PointRend models.

## Installation
We implements PointRend under the [mmdetection]() framework, note that the mmdetection version here we used is 1.1.0.
We follow the official mmdetection [installation](), and list our experiment environments in *requirements.txt*.

## Prepare Dataset
Download the dataset from official [challenge site](), and then put it under the folder *mmdet_furniture/data/*, 
the dataset folder structure in our experiments looks like this:
```python
mmdet_furniture
|-- data
     |-- future
           |-- annotations
                 |-- test_set.json 
                 |-- train_set.json  
                 |-- val_set.json
           |-- images
                 |-- train
                 |-- val
                 |-- test
```

## Train & Evaluation
### Train
To train a PointRend model on the 3DFuture dataset, run:
```python
# train with 8 gpus using one machine:
./tools/dist_train.sh furniture_config/pointrend/CONFIG_NAME.py 8 

# We list 5 config files with different training settings under folder furniture_config/pointrend/, 
# which are used for this competition. For instance, you can run one of the configs like this:
./tools/dist_train.sh furniture_config/pointrend/pointrend_x101_64x4d_dcn_fpn_fp16_p2p6.py 8 
```
After training finished, the trained model and training log will be saved under the folder *work_dirs/CONFIG_NAME*

### Evaluation
We follow
## Performance
### Single Model
### Model Ensemble
