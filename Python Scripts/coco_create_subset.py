import tarfile
import os, shutil
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

coco_p = Path('./output/coco_instances.json').absolute()
with open(coco_p) as f:
    coco_json = json.load(f)

# "lookup" lists to expedite creation of coco_json dicts for image subsets
# Look up image id's by image filename
img_ids = [coco_json.get('images')[i].get('id') for i in range(len(coco_json['images']))]
img_files = [coco_json.get('images')[i].get('file_name') for i in range(len(coco_json['images']))]
img_lookup = dict(zip(img_files, img_ids))
# Look up annotations by image id's
from itertools import groupby
img_ann = [coco_json.get('annotations')[i].get('image_id') for i in range(len(coco_json['annotations']))]
_ = list(enumerate(sorted(img_ann)))
ann_lookup = dict()
for k, v in groupby(_, lambda x:x[1]):
    ann_lookup[k] = [v[0] for v in list(v)]


from collections import defaultdict
# Create new coco_json for the smaller set of images
def coco_sub_json(img_files, coco_json=coco_json):
    '''
    Takes a set of images in a list and creates a smaller corresponding coco_json dictionary for the subset from the main coco_instance json file.

    img_files = list of images for which to create the new coco_json dict

    coco_json = original (large) dict from main coco_instances.json
    '''
    new_dict = defaultdict().fromkeys(list(coco_json.keys()))
    new_dict['info'] = coco_json.get('info').copy()
    new_dict['licenses'] = coco_json.get('licenses').copy()
    new_dict['categories'] = coco_json.get('categories').copy()
    new_dict['images'] = []
    ann = coco_json.get('annotations')
    new_dict['annotations'] = []
    
    for file in img_files:
        idx = img_lookup.get(file)
        new_dict['images'].append(coco_json.get('images')[idx].copy())
        for ann_id in ann_lookup[idx]:
            new_dict['annotations'].append(ann[ann_id])
    
    return new_dict

img_path = Path("./output/images/").absolute()
images = os.listdir(img_path)
fish_subset = [img_name for img_name in np.random.choice(images, size=300, replace=False)]
sub_train = fish_subset[:250]
sub_val = fish_subset[250:]
# sub_test = fish_subset[8000:]

src = img_path
dest_tr, dest_v = ("./subset0/{}/".format(x) for x in ["train", "val"])
# src = img_path
# dest_tr, dest_v, dest_ts = ("./subset2/{}/".format(x) for x in ["train", "val", "test"])

for item in [(dest_tr,sub_train),(dest_v,sub_val)]:
    os.makedirs(item[0] + "images/")
    for img in item[1]:
        shutil.copy(str(src) + "/" + img, item[0] + "images/" + img)
    with open(item[0] + 'coco_instances.json', 'w') as f:
        json.dump(coco_sub_json(item[1]), f)
# for item in [(dest_tr,sub_train),(dest_v,sub_val), (dest_ts,sub_test)]:
#     os.makedirs(item[0] + "images/")
#     for img in item[1]:
#         shutil.copy(str(src) + "/" + img, item[0] + "images/" + img)
#     with open(item[0] + 'coco_instances.json', 'w') as f:
#         json.dump(coco_sub_json(item[1]), f)
