import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm


# 일단은 사용하지 않고 있습니다.

def get_img_stats(img_dir, img_ids):
    img_info = dict(means=[], stds=[])
    for img_id in tqdm(img_ids, desc = "img_id"):
        for path in glob(os.path.join(img_dir, img_id, '*')):
            print(path)
            img = np.array(Image.open(path))
            img_info['means'].append(img.mean(axis=(0,1)))
            img_info['stds'].append(img.std(axis=(0,1)))
    return img_info['means'], img_info['stds']

