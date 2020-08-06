'''
Script for testing out the COCO API.
====================================

The COCO API is developed and maintained by developers of 
the COCO dataset and offers tools for interacting with it.
'''

# `pycocotools` is the library for the Python API
# The library can be installed from the conda-forge repo.
from pycocotools.coco import COCO

# other libraries needed to work with the data
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

# I have the COCO dataset stored on an external hard drive
# so the following is also useful:
from pathlib import PurePath
p = PurePath("/media/tamer/Samsung_T5/COCO_data")

# Define the directory. Annotations is used as an example.
dataDir=str(p)
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
# Initialize the api and index the annotations
coco=COCO(annFile)

# Display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
s_nms = set([cat['supercategory'] for cat in cats])
catTtl = "COCO Categories"
scatTtl = "COCO Supercategories"
print(catTtl+':', '\n'+'-'*len(catTtl), '\n{}\n'.format('; '.join(nms)))
print(scatTtl+':', '\n'+'-'*len(scatTtl), '\n{}\n'.format('; '.join(s_nms)))

# Categories can be filtered by supercategory, category or category ID
# Use `getCatIds()` method and enter a list of the object categories
# to search for in images. For example, enter [person, sports, tennis 
# racket] for images tagged with all three items.

# Return the category ID's for labels:
catIds = coco.getCatIds(catNms=['person','horse','cow'])
# Get the image ID's with those labels
imgIds = coco.getImgIds(catIds=catIds )
# Select a random image from the returned images
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
# Use scikit-image and pyplot to read the image and display it.
# From a local directory:
Img_local = io.imread('{}/{}/{}'.format(dataDir, dataType, img['file_name']))
# From the web with URL:
Img_url = io.imread(img['coco_url'])

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
plt.axis('off')
plt.imshow(Img_local)
# plt.imshow(Img_url)
plt.show()

# load and display instance annotations
plt.imshow(Img_local); plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)