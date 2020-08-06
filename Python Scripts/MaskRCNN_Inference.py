'''
Python script to run inference on trained weights
for fish dataset
'''

# Run first if using CNN on GPU with Tensorflow 1.xx!
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)
# Import standard libraries
import os
import sys
import json
import time
# Import python external libraries
import numpy as np
from PIL import Image, ImageDraw
# Set root directory - should be 
ROOT_DIR = '/home/tamer/UChiMSCA/MSCA37011_DeepLearningImgRec/Project/MaskRCNN_model/Mask_RCNN/'

# Import mrcnn libraries
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class FishConfig(Config):
    """Configuration for training on the fish dataset.
    Derives from the base Config class and overrides values specific
    to the fish dataset.
    """
    # Name of configuration
    NAME = "fish_img"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (fish)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # This is equivalent to batch size
    # Higher means more images are trained per epoch
    STEPS_PER_EPOCH = 500
    # This is how often validation is run
    # Adds extra time cost
    VALIDATION_STEPS = 100
    # Chooose backbone
    BACKBONE = 'resnet101'
    # Learning rate and convergence parameters
    # re-adjust at end of epoch for better convergence
    LEARNING_RATE = 0.00001 # 0.001 default
    LEARNING_MOMENTUM = 0.9
    # Regularization
    WEIGHT_DECAY = 0.0001
    # NMS threshholds
    DETECTION_NMS_THRESHOLD = 0.2
    RPN_NMS_THRESHOLD = 0.5  # 0.7 default
    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_NMS_THRESHOLD = 0.7
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 40
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000
    
    # OPTIMIZER is a new parameter that was not
    # in the original Matterport implementation 
    # (I added it to `model.py`).
    # Allows adding different optimizers.
    # It will only work with the modified `model.py` 
    # script in this directory!!
    OPTIMIZER = 'SGD'
    
config = FishConfig()
config.display()

# *********************** Define COCO-like dataset ******************

class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        
        # Add the class names using the base method from utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
                
    def load_mask(self, image_id):
        """ Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids

# ***************************** DATASETS **************************** 

dataset_test = CocoLikeDataset()
dataset_test.load_data('../test_images/coco_instances.json', '../test_images/images')
dataset_test.prepare()

dataset_val = CocoLikeDataset()
dataset_val.load_data('./subset_A/val/coco_instances.json', './subset_A/val/images')
dataset_val.prepare()

dataset = dataset_test
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

# *************************** INFERENCE *****************************

# Set up inference configuration
class InferenceConfig(FishConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85

inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.abspath("./Mask_RCNN_weights/Run8_20200607/mask_rcnn_fish_img_0012.h5")
# model_path = os.path.join(ROOT_DIR, "logs/fish_img20200607T1310/mask_rcnn_fish_img_0018.h5")
# model_path = model.find_last()  # last trained

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

import skimage
real_test_dir = './real_test3'
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in image_paths[:5]:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names, r['scores'], figsize=(5,5))


# Specify path to test images
from skimage import io
test_dir = '../test_images/images'
image_paths = []
for filename in os.listdir(test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg']:
        image_paths.append(os.path.join(test_dir, filename))


# Import functions to calculate evaluation metrics
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt, mold_image

tr = test_results

def eval_model(dataset=dataset, cfg=inference_config):
    APs = list()
    precision_vals = list()
    recall_vals = list()
    overlaps = list()

    for idx in dataset.image_ids:
        image, shape, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, inference_config, idx, use_mini_mask=False)

        AP, prec, recall, overlap = compute_ap(gt_bbox, gt_class_id, gt_mask, tr[idx][0]["rois"], tr[idx][0]["class_ids"], tr[idx][0]["scores"], tr[idx][0]['masks'])

        APs.append(AP)
        precision_vals.append(prec)
        recall_vals.append(recall)
        overlaps.append(overlap)

    return APs, precision_vals, recall_vals, overlaps

fish_eval =  eval_model()

APs, Precisions, Recalls, Overlaps = fish_eval[0], fish_eval[1], fish_eval[2], fish_eval[3]


import matplotlib.pyplot as plt
from mrcnn.visualize import plot_precision_recall, plot_overlaps

im_ = np.random.choice(range(len(APs)))
plot_precision_recall(APs[im_], Precisions[im_], Recalls[im_])

avPr_vals = []
avRec_vals = []
for im_idx in range(len(APs)):
    avPr = np.mean(Precisions[im_idx]/len(Precisions[im_idx]))
    avRec = np.mean(Recalls[im_idx]/len(Recalls[im_idx]))
    avPr_vals.append(avPr)
    avRec_vals.append(avRec)

plot_precision_recall(np.mean(APs), avPr_vals, avRec_vals)

