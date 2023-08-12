import torch
import numpy as np
import random
from typing import List
import cv2
import tqdm


def measure_distances(features_a, features_b):
    distances = torch.cdist(torch.permute(features_a,[1,0]),torch.permute(features_b,[1,0]))
    return distances

def super_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)

def test_for_positional_class_transpose(imgs):
    average = np.mean(np.array(imgs),axis=0,keepdims=False)
    #average_f = np.flip(np.flip(average, (0)), (1))
    average_f = np.transpose(average, (1,0,2))
    return np.mean(np.square(average.astype(np.float16)-average_f.astype(np.float16)))

def test_for_positional_class_flips(imgs):
    average = np.mean(np.array(imgs),axis=0,keepdims=False)
    average_f = np.flip(np.flip(average, (0)), (1))
    #average_f = np.transpose(average, (1,0,2))
    return np.mean(np.square(average.astype(np.float16)-average_f.astype(np.float16)))

def align_images(seed, images, masks, quite=True):
    c_x = seed.shape[0]//2
    c_y = seed.shape[1]//2
    image_size = (seed.shape[0],seed.shape[1])
    r_mat = [ cv2.getRotationMatrix2D((c_x,c_y), x,  1.0) for x in range(360)]
    proposed_data_corrupted_images = []
    proposed_used_test_masks        = []
    for k, image in enumerate(tqdm.tqdm(images, ncols=100, desc = 'Rotating', disable=quite)):
        rotation_ideal = []
        test_img = seed.astype(np.float16)[c_x-(c_x//2):c_x+(c_x//2), c_y-(c_y//2):c_y+(c_y//2)]
        for x in range(0,360):
            candidate = cv2.warpAffine(image, r_mat[x], image_size).astype(np.float16)[c_x-(c_x//2):c_x+(c_x//2), c_y-(c_y//2):c_y+(c_y//2)]
            rotation_ideal.append(np.mean(np.square(test_img-candidate )))
        proposed_data_corrupted_images.append(cv2.warpAffine(image, r_mat[np.argmin(rotation_ideal)], image_size))
        if not masks is None:
            masks_rounded = cv2.warpAffine(masks[k], r_mat[np.argmin(rotation_ideal)], image_size)
            masks_rounded[masks_rounded>128]  = 255
            masks_rounded[masks_rounded<=128] = 0
        else:
            masks_rounded = None
        proposed_used_test_masks.append(masks_rounded)
    
    return proposed_data_corrupted_images, proposed_used_test_masks

def positional_test_and_alignment(images: List[np.ndarray], threashold: float, masks: List[np.ndarray]=None, align: bool = True, quite: bool = True):
    if test_for_positional_class_transpose(images) < threashold:
        if align: # Speedup trick here just becasue this is deterministic for classes
            proposed_data_corrupted_images, proposed_used_test_masks =  align_images(images[0], images, masks, quite=quite)
            if test_for_positional_class_transpose(proposed_data_corrupted_images) < threashold:
                return False, images, masks, False
            else:
                return True, proposed_data_corrupted_images, proposed_used_test_masks, True  
        return False, images, masks, False 
    return True, images, masks, False

