# prepare environment
virtualenv -p python3 venv
source venv/bin/activate

pip install torch==1.4.0
# pip install /mnt/truenas/scratch/lqf/software/torch-1.4.0-cp36-cp36m-manylinux1_x86_64.whl
pip install torchvision==0.5.0

cd mmdet_furniture
pip install -v -e .

pip install mmcv==0.5.0
pip install numpy==1.16
pip install cython==0.29.21
pip install scipy==1.5.2
pip install fvcore==0.1.1.post20200716

git clone https://github.91chifun.workers.dev//https://github.com/zehuichen123/cocoapi.git
cd cocoapi/PythonAPI
python setup.py install
cd ../..
rm -r cocoapi

# download models
mkdir infer_results         # path to save inference results(pkl)

 # infer 5 models
./tools/dist_test.sh furniture_config/pointrend/pointrend_res2net101_dcn_fpnbfp_fp16_p2p6_enrichfeat.py \
    work_dirs/pointrend_res2net101_dcn_fpnbfp_fp16_p2p6_enrichfeat/epoch_44.pth $GPUS \
    --out infer_results/pointrend_res2net101_dcn_fpnbfp_fp16_p2p6_enrichfeat.pkl

./tools/dist_test.sh furniture_config/pointrend/pointrend_res2net101_dcn_fpnbfp2repeat_p2p6_fp16_enrichfeat_largeboxalign.py \
    work_dirs/pointrend_res2net101_dcn_fpnbfp2repeat_p2p6_fp16_enrichfeat_largeboxalign/epoch_44.pth $GPUS \
    --out infer_results/pointrend_res2net101_dcn_fpnbfp2repeat_p2p6_fp16_enrichfeat_largeboxalign.pkl

./tools/dist_test.sh furniture_config/pointrend/pointrend_x101_64x4d_dcn_fpn_fp16_p2p6.py \
    work_dirs/pointrend_x101_64x4d_dcn_fpn_fp16_p2p6/epoch_44.pth $GPUS \
    --out infer_results/pointrend_x101_64x4d_dcn_fpn_fp16_p2p6.pkl

./tools/dist_test.sh furniture_config/pointrend/pointrend_x101_64x4d_dcn_fpnbfp_fp16_p2p6_enrichfeat.py \
    work_dirs/pointrend_x101_64x4d_dcn_fpnbfp_fp16_p2p6_enrichfeat/epoch_44.pth $GPUS \
    --out infer_results/pointrend_x101_64x4d_dcn_fpnbfp_fp16_p2p6_enrichfeat.pkl

./tools/dist_test.sh furniture_config/pointrend/pointrend_x101_64x4d_dcn_fpnbfp_fp16_p2p6_lr001.py \
    work_dirs/pointrend_x101_64x4d_dcn_fpnbfp_fp16_p2p6_lr001/epoch_44.pth $GPUS \
    --out infer_results/pointrend_x101_64x4d_dcn_fpnbfp_fp16_p2p6_lr001.pkl

# ensemble results
cd ../code_furniture/cython_nms
python setup.py install
cd ..
ln -s ../mmdet_furniture/infer_results ./infer_results
ln -s ../mmdet_furniture/data ./data
python group_ensemble.py
zip segmentation_resutls.json.zip segmentation_resutls.json