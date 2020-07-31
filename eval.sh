# prepare environment
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt

cd mmdet_furniture
# download models
mkdir save_models           # path to save pretrain models
mkdir infer_results         # path to save inference results(pkl)
wget -P save_models/xxx https://....

# infer 5 models
./tools/dist_test.sh furniture_config/pointrend_res2net101_dcn_fpnbfp_fp16_p2p6_enrichfeat.py \
    save_models/pointrend_res2net101_dcn_fpnbfp_fp16_p2p6_enrichfeat.pth num_GPU \
    infer_results/pointrend_res2net101_dcn_fpnbfp_fp16_p2p6_enrichfeat.pkl


# ensemble results
cd ../code/cython_nms
python setup.py install
cd ..
ln -s ../mmdet_furniture/infer_results ./results
ln -s ../mmdet_furniture/data ./data
python group_ensemble.py
zip segmentation_resutls.json.zip segmentation_resutls.json