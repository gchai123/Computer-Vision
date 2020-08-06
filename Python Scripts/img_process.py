from skimage import io
from skimage import color, filters, feature, exposure
from matplotlib import pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

file = '/home/tamer/UChiMSCA/MSCA37011_DeepLearningImgRec/Project/fish_img/00001.png/'

img = io.imread(file)
plt.imshow(img)

# img = exposure.equalize_hist(img)
# plt.imshow(img)
# edges = filters.roberts(color.rgb2gray(img))
# img = exposure.equalize_adacoordhist(img)
# plt.imshow(img)
# plt.imshow(edges)

img_gs = color.rgb2grey(img)
plt.imshow(img_gs)
img_hsv = color.rgb2hsv(img)
plt.imshow(img_hsv, cmap=plt.cm.hsv)

# Separate the hue, saturation and value channels
img_hue = img_hsv[:,:,0]
img_sat = img_hsv[:,:,1]
img_val = img_hsv[:,:,2]

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 8))
ax0.imshow(img_hue)
ax0.set_title("Hue channel")
ax0.axis('off')
ax1.imshow(img_sat)
ax1.set_title("Saturation channel")
ax1.axis('off')
ax2.imshow(img_val)
ax2.set_title("Value channel")
ax2.axis('off')

fig.tight_layout()

_= plt.hist(img_sat.ravel(), bins=256)

img_adj = np.vectorize(lambda x: x+0.5 if x<0.5 else x-0.5)
adjusted = img_adj(img_sat)
plt.imshow(adjusted)

_= plt.hist(adjusted.ravel(), bins=256)


# Examine thresholding ocoordions
filters.try_all_threshold(adjusted, verbose=False)

# thresh = filters.threshold_niblack(img_hsv, window_size=25)
# img_thresh = np.where(img_hsv[:,:,0] > thresh[:,:,0], 255, 0)
# plt.imshow(img_thresh)

th_1 = filters.threshold_otsu(adjusted)
img_th1 = adjusted < th_1
plt.imshow(img_th1)

th_2 = filters.threshold_sauvola(img_hsv, window_size=7)
img_th2 = img_hsv < th_2
plt.imshow(img_th2)

# combine results of both filters
img_bin = np.logical_or(img_th1, img_th2)
plt.imshow(img_bin)


# Use erosion to filter out noise (dots) then apply median filter to smooth and isolate regions
from skimage.morphology import area_closing, disk
from skimage.filters import median
from scipy import ndimage as ndi

img_bin = area_closing(img_bin, area_threshold=128)
img_bin = median(img_bin, disk(5))
# img_bin = ndi.binary_fill_holes(img_bin)
plt.imshow(img_bin)



# ============================================================================ #

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb


image = data.coins()[50:-50, 50:-50]

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, square(3))

# remove artifacts connected to image border
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

for region in regionprops(label_image):
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

ax.set_axis_off()
plt.tight_layout()
plt.show()

# ============================================================ #

from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import os

img_dir = Path('/home/tamer/UChiMSCA/MSCA37011_DeepLearningImgRec/Project/fish_img')
msk_dir = Path('/home/tamer/UChiMSCA/MSCA37011_DeepLearningImgRec/Project/fish_mask')
fishImg = os.listdir(img_dir)
fishMask = os.listdir(msk_dir)

images = [im for im in (fishImg and fishMask) if im.endswith(".png")]

imgFile = np.random.choice(images)
# imgFile = "00071.png"
img, mask = (Image.open(str(i) + '/' + imgFile) for i in (img_dir, msk_dir))
img, mask = img.convert('RGB'), mask.convert('RGB')

from itertools import product
w, h = mask.size  # width, height of mask
coords = list(product(range(w), range(h)))
isolated_masks = dict()
for coord in coords:
    pixelRGB = mask.getpixel(coord)
    pixelRGB_str = str(mask.getpixel(coord))

    if not pixelRGB == (0,0,0):
        if isolated_masks.get(pixelRGB_str) is None:
            isolated_masks[pixelRGB_str] = Image.new('1', (w+2, h+2))

        isolated_masks[pixelRGB_str].putpixel((coord[0]+1, coord[1]+1), 1)

from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPolygon

# objs = dict()
for key, mask in isolated_masks.items():
    contours = find_contours(mask, 0.5, positive_orientation='low')
    # objs[key] = contours
    polygons = []
    for contour in contours:
        # flip row, col coord's to to x,y and subtract padding
        row, col = zip(*contour)
        contour = np.column_stack((np.array(col) - 1, np.array(row) - 1))
        
        # Convert contour to simple polygon
        poly = Polygon(contour)
        poly = poly.simplify(0.8, preserve_topology=False)

        if (poly.area > 16): # Ignore tiny polygons
            if (poly.geom_type == 'MultiPolygon'):
                # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
                poly = poly.convex_hull

            if (poly.geom_type == 'Polygon'): # Ignore if still not a Polygon (could be a line or point)
                polygons.append(poly)

        


# fig, ax = plt.subplots()
# plt.imshow(img)
# for contour in objs.values():
#     for i in range(len(contour)):
#         ax.plot(contour[i][:, 1], contour[i][:, 0], linewidth=1)


# ============================================================================ #

# from itertools import repeat
# import json

# im = (Image.open(str(msk_dir)+ "/" + imgFile)).convert('RGB')
# im = np.asarray(im)

def colorCatExtr(img:np.array):
    '''
    Extract unique colors from mask image to identify color categories.
    -------------------------------------------------------------------
    img: RGB array of image
    Returns mask color values cast as strings; e.g. yellow is returned as "(255, 255, 0)"
    '''

    color_cats = []
    colors = (np.unique(img.reshape(-1, img.shape[2]), axis=0)).tolist()
    for col in colors:
        # black is background so ignore
        if not col == [0, 0, 0]:
            col_id = str(tuple(col))
            color_cats.append(col_id)
    return color_cats

def mask_definitions(images, super_category="animal", category="fish"):
    '''
    Creates mask definitions from mask images based on color categories.
    Returns a .json object to be saved in directory containing both image and mask files.
    '''
    import json
    from itertools import repeat
        
    super_categories = dict(super_category=category)
    masks = dict()
    
    for img in images:
        mask = (Image.open(str(msk_dir)+ "/" + img)).convert('RGB')
        mask = np.asarray(mask)
        clr_ids = colorCatExtr(mask)
        color_categories = dict(zip(clr_ids, repeat({"category": "fish", "super_category": "animal"}, len(clr_ids))))
        masks["images/" + img] = dict(mask="masks/" + img, color_categories=color_categories)
    
    return json.dumps({"masks":masks, "super_categories":super_categories}, indent=4)

with open("mask_definitions.json", "w") as f:
    f.write(mask_definitions(images))

# import json
# print(json.dumps(masks, sort_keys=False, indent=4))


# ---------------------------------------------------------------------------- #
import numpy as np
import random
from PIL import Image, ImageEnhance

fg1 = Image.open("../input/foregrounds/animal/fish/00345.png")
fg2 = Image.open("../input/foregrounds/animal/fish/00361.png")
fg3 = Image.open("../input/foregrounds/animal/fish/00370.png")
bg_path = "../input/backgrounds/00005.png"
fgs = [fg1, fg2, fg3]

def transform_foreground(fg_image):
        # Open foreground and get the alpha channel
        # fg_image = Image.open(fg_path)
        fg_alpha = np.array(fg_image.getchannel(3))
        assert np.any(fg_alpha == 0), f'foreground needs to have some transparency'

        # ** Apply Transformations **
        # Rotate the foreground
        angle_degrees = random.randint(-30, 30)
        fg_image = fg_image.rotate(angle_degrees, resample=Image.BICUBIC, expand=0)

        # Scale the foreground
        # scale = random.random() * .5 + .5 # Pick something between .5 and 1
        # new_size = (int(fg_image.size[0] * scale), int(fg_image.size[1] * scale))
        # fg_image = fg_image.resize(new_size, resample=Image.BICUBIC)

        # Adjust foreground brightness
        brightness_factor = random.random() * .4 + .7 # Pick something between .7 and 1.1
        enhancer = ImageEnhance.Brightness(fg_image)
        fg_image = enhancer.enhance(brightness_factor)

        # Add any other transformations here...
        fg_area = np.sum(np.logical_not(np.array(fg_image)[...,3] == 0))
        return fg_image, fg_area

def compose_images(foregrounds, background_path, width=512, height=512):
        
        # Open background and convert to RGBA
        background = Image.open(background_path)
        background = background.convert('RGBA')

        # Crop background to desired size (width x height), randomly positioned
        bg_width, bg_height = background.size
        max_crop_x_pos = bg_width - width
        max_crop_y_pos = bg_height - height
        assert max_crop_x_pos >= 0, f'desired width, {width}, is greater than background width, {bg_width}, for {str(background_path)}'
        assert max_crop_y_pos >= 0, f'desired height, {height}, is greater than background height, {bg_height}, for {str(background_path)}'
        crop_x_pos = random.randint(0, max_crop_x_pos)
        crop_y_pos = random.randint(0, max_crop_y_pos)
        composite = background.crop((crop_x_pos, crop_y_pos, crop_x_pos + width, crop_y_pos + height))
        composite_mask = Image.new('RGB', composite.size, 0)

        for fg in fgs:

            # Perform transformations
            fg_image, fg_area = transform_foreground(fg)

            # Prevent any single fg image from crowding the composite image by scaling it down if it is large w.r.t. the frame
            bg_area = bg_width*bg_height

            if fg_area > 0.3*bg_area/len(foregrounds):
                scale = (0.3/len(foregrounds))*(bg_area/fg_area)
                new_size = ( int(scale * fg_image.size[0]), int(scale * fg_image.size[1]) )
                fg_image = fg_image.resize(new_size, resample=Image.BICUBIC)
            
            # Choose a random x,y position for the foreground
            max_x_position = composite.size[0] - fg_image.size[0]
            max_y_position = composite.size[1] - fg_image.size[1]
            assert max_x_position >= 0 and max_y_position >= 0, \
            f'foreground {fg} is too big ({fg_image.size[0]}x{fg_image.size[1]}) for the requested output size ({width}x{height}), check your input parameters'
            paste_position = (random.randint(0, max_x_position), random.randint(0, max_y_position))

            # Create a new foreground image as large as the composite and paste it on top
            new_fg_image = Image.new('RGBA', composite.size, color = (0, 0, 0, 0))
            new_fg_image.paste(fg_image, paste_position)

            # Extract the alpha channel from the foreground and paste it into a new image the size of the composite
            alpha_mask = fg_image.getchannel(3)
            new_alpha_mask = Image.new('L', composite.size, color = 0)
            new_alpha_mask.paste(alpha_mask, paste_position)
            composite = Image.composite(new_fg_image, composite, new_alpha_mask)

            # Grab the alpha pixels above a specified threshold
            alpha_threshold = 200
            mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
            uint8_mask = np.uint8(mask_arr) # This is composed of 1s and 0s

            # Multiply the mask value (1 or 0) by the color in each RGB channel and combine to get the mask
            mask_rgb_color = [255, 0, 0]
            red_channel = uint8_mask * mask_rgb_color[0]
            green_channel = uint8_mask * mask_rgb_color[1]
            blue_channel = uint8_mask * mask_rgb_color[2]
            rgb_mask_arr = np.dstack((red_channel, green_channel, blue_channel))
            isolated_mask = Image.fromarray(rgb_mask_arr, 'RGB')
            isolated_alpha = Image.fromarray(uint8_mask * 255, 'L')

            composite_mask = Image.composite(isolated_mask, composite_mask, isolated_alpha)

        return composite, composite_mask