# 1st Place Solutions of 3D AI Challenge 2020 - Instance Segmentation Track

This repository maintains our solution to [3D AI Challenge Instance Segmentation Track](), which ranks 1st in both trackA (validation set)] and trackB (test set). 
Our solution is a weighted ensemble of several [PointRend]() models. This repository implements PointRend and train/evaluates it on the [3DFuture dataset](), 
based on the open projects [mmdetection]() and [detectron2 PointRend](). We also provide ensemble code on our trained PointRend models.

## Installation
We implements PointRend under the [mmdetection]() framework, note that the mmdetection version here we used is 1.1.0.
We follow the official mmdetection [installation](), and list our experiment environments in *requirements.txt*.

## Prepare Dataset
Download the dataset from official [challenge site](), and the put it under the folder *mmdet_furniture/data/*, 
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

## Training & Evaluation

## Performance
### Single Model
### Model Ensemble
