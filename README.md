# 1st Place Solutions of 3D AI Challenge 2020 - Instance Segmentation Track

This repository maintains our solution to [3D AI Challenge Instance Segmentation Track](), which ranks 1st in both trackA (validation set)] and trackB (test set). 
Our solution is a weighted ensemble of several [PointRend]() models. This repository implements PointRend and train/evaluates it on the [3DFuture dataset](), 
based on the open projects [mmdetection]() and [detectron2 PointRend](). We also provide ensemble code on our trained PointRend models.

## Environment Requirements

CUDA 10.0, pytorch 1.4.0, torchvision 0.5.0

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

## Only two lines to get results

We provide script which allows you to get our final submission:
```bash
export num_GPU = 8      # num of GPU you have
source eval.sh
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
We provide five pretrained PointRend models which can be used for test generation. Download the pretrained weights from [here](),
and put it under folder *work_dirs*.

### Evaluation
Two steps for generating final submission file format on test dataset.
First generate infer results in the pkl format based on the trained model.
Then convert data from pkl to json format which is then zipped and used for submit the test server. 

```python
# Firstly, evaluate on test set and generate infer results with pkl format:
./tools/dist_test.sh furniture_config/pointrend/CONFIG_NAME.py work_dirs/CONFIG_NAME/epoch_44.pth 8 --out out_results/CONFIG_NAME/segmentation_resutls.pkl
# Note we train PointRend for 44 epochs and defaultly pretrained weights of the last epoch are used for infer.
```
Above scripts will generate a *segmentation_resutls.pkl* file under the folder *out_results/CONFIG_NAME*.

```python
# Then, convert from pkl to json format:
python tools/submit_test.py --root ROOT_PATH --config_name CONFIG_NAME  # ROOT_PATH: absolute file path for project mmdet_furniture 
```
Above scripts will generate a *segmentation_resutls.json* under the folder *out_results/CONFIG_NAME*


```python
# Finally, zip the json file and submit to test server
zip -q segmentation_resutls.zip segmentation_resutls.json
```
Above scripts will generate a *segmentation_resutls.zip* under the folder *out_results/CONFIG_NAME*, which can be used for submitting test server.

We also provides five *segmentation_resutls.zip* files which we used as single model for submitting test server, download from [here]().

## Performance
### Single Model
We report PointRend performance under the test dataset with multi-scale train/test. All the models are trained using 8X 2080Ti.
#### PointRend
Backbone | FP16 | Mask AP | AP50 | AP75 | APs | APm | APl | config| pretrained |
--------- | --------- | ---------- | ---------| ----------| ----------| ---------| -----------| -----------| -----------
X101_64x4d|yes|**77.38**|89.34|83.28|45.31|	71.21|**82.24**|pointrend_x101_64x4d_dcn_fpn_fp16_p2p6|[link]()
X101_64x4d|yes|77.32|89.79|83.24|45.78|	72.25|81.7|pointrend_x101_64x4d_dcn_fpnbfp_fp16_p2p6_lr001|[link]()
X101_64x4d|yes|**77.37**|89.78|**83.39**|46.07|**72.84**|81.68|pointrend_x101_64x4d_dcn_fpnbfp_fp16_p2p6_enrichfeat|[link]()
res2net101|yes|76.95|89.92|82.97|45.5|72.49|81.71|pointrend_res2net101_dcn_fpnbfp2repeat_p2p6_fp16_enrichfeat_largeboxalign|[link]()
res2net101|yes|77.21|**90.09**|82.88|**47.3**|71.98|81.97|pointrend_res2net101_dcn_fpnbfp_fp16_p2p6_enrichfeat|[link]()


### Model Ensemble

Before you run ensemble code, you need to install Soft-NMS implementation borrowed from [SimpleDet].
```bash
cd code/cython_nms
python setup.py install
```

We adopted two different ways for model reweight. A detail description please refer to our report.

Method   | Mask AP | AP50 | AP75 | APs | APm | APl |
---------| ------- | -----| -----| ----| ----| ----|
Linear-Reweight | 78.92 | 91.56 | 85.02 | 48.59 | 73.69 | 83.93 |
Linear-interplot | **79.03** | 91.64 | 85.09 | 48.42 | 74.09 | 84.04|

