import mmcv
import matplotlib.pyplot as plt
import numpy as np

future_path = '/mnt/truenas/scratch/czh/data/future/'
future_train_path = future_path + 'annotations/train_set.json'
future_train_data = mmcv.load(future_train_path)
future_train_anno = future_train_data['annotations']

coco_path = '/mnt/truenas/scratch/czh/data/coco/'
coco_train_path = coco_path + 'annotations/instances_train2017.json'
coco_train_data = mmcv.load(coco_train_path)
coco_train_anno = coco_train_data['annotations']

print("Loading Annotations Done")

def compute_area(anno_data):
    area_list = []
    rand_ind = np.random.choice(len(anno_data), 50000, replace=False)
    for ind in rand_ind:
        each_anno = anno_data[ind]
        x, y, w, h = each_anno['bbox']
        area_list.append((w * h) ** 0.5)
    area_data = np.array(area_list)
    return area_data

future_area_data = compute_area(future_train_anno)
coco_area_data = compute_area(coco_train_anno)

fig = plt.figure(figsize=(12, 5))
plt.style.use('seaborn')
plt.hist(future_area_data, range=(0, 900), bins=100, color='red', alpha=0.5, label='3D-FUTURE')
plt.hist(coco_area_data, range=(0, 900), bins=100, color='blue', alpha=0.5, label='COCO')
plt.legend()
plt.xlim(0, 800)
plt.xlabel('Sqrt of Instance Area')
plt.ylabel('Number of Instances')
plt.grid(True)
fig.savefig('fig/size_distribution.png', dpi=200)

print("Plot Figure Done")

