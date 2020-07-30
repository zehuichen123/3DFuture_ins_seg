# 1st Place Solutions of 3D AI Challenge 2020 - Instance Segmentation Track

This repository maintains our solution to [3D AI Challenge Instance Segmentation Track](), which ranks 1st in both trackA (validation set)] and trackB (test set). 
Our solution is a weighted ensemble of several [PointRend]() models. This repository implements PointRend and train/evaluates it on the [3DFuture dataset](), 
based on the open projects [mmdetection]() and [detectron2 PointRend](). We also provide ensemble code on our trained PointRend models.

## Installation
We implements PointRend under the [mmdetection]() framework, note that the mmdetection version here we used is 1.1.0.
We follow the official mmdetection [installation](), and list our experiment environments in *requirements.txt*.

## Prepare Dataset
Download the dataset from official [challenge site](), and the put it under the folder *mmdet_furniture/data/*, 
the dataset folder structure looks like this:
```python
|--- mmdet_furniture
# train
mot17_data=['MOT17-02-FRCNN','MOT17-05-FRCNN','MOT17-09-FRCNN','MOT17-10-FRCNN','MOT17-13-FRCNN']
mot15_data=['KITTI-17','ETH-Sunnyday','ETH-Bahnhof','PETS09-S2L1','TUD-Stadtmitte']

# validation
mot17_data_eval=['MOT17-11-FRCNN', 'MOT17-04-FRCNN']
```

## Training & Evaluation

## Performance
### Single Model
### Model Ensemble
