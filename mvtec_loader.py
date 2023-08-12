import cv2 
import numpy as np
import os
import random

def load_corrupted_data(class_name: str, 
                        data_dir: str, 
                        num_corrupted: int, 
                        size: tuple = (256,256), 
                        crop_size: tuple = (224,224)):
    train_images = load_training_data(class_name, data_dir, size=size, crop_size=crop_size)
    test_images, test_masks, corr_types = load_testing_data(class_name, data_dir, size=size, crop_size=crop_size)
    ziped = list(zip(test_images, test_masks, corr_types))
    random.shuffle(ziped)
    test_images, test_masks, corr_types = zip(*ziped)
    test_images = list(test_images)
    test_masks = list(test_masks)
    corr_types = list(corr_types)
    return train_images + test_images[:num_corrupted], [np.zeros_like(test_masks[0]) for x in range(len(train_images))] + test_masks[:num_corrupted], corr_types

def load_testing_data(class_name: str, 
                      data_dir: str, 
                      size: tuple = (256,256), 
                      crop_size: tuple = (224,224)):
    assert class_name in os.listdir(data_dir)
    img_dir = data_dir+class_name+'/test/'
    ann_dir = data_dir+class_name+'/ground_truth/'
    test_images = []
    test_truths = []
    test_class = []
    x = int(size[0]/2- crop_size[0]/2)
    y = int(size[1]/2- crop_size[1]/2)
    for directory in os.listdir(img_dir):
        for filename in os.listdir(img_dir+directory+'/'):
            test_images.append(cv2.resize(cv2.imread(img_dir+directory+'/'+filename),size)[x:x+crop_size[0],y:y+crop_size[1]])
            if directory != 'good':
                test_truths.append(cv2.resize(cv2.imread(ann_dir+directory+'/'+filename[:-4]+'_mask.png'),size)[x:x+crop_size[0],y:y+crop_size[1]])
            else:
                test_truths.append(np.zeros_like(test_images[-1]))
            test_class.append(directory)
    return test_images, test_truths, test_class

def load_training_data(class_name: str, 
                      data_dir: str, 
                      size: tuple = (224,224), 
                      crop_size: tuple = (224,224)):
    assert class_name in os.listdir(data_dir)
    train_images = []
    dir = data_dir+class_name+'/train/good/'
    x = int(size[0]/2- crop_size[0]/2)
    y = int(size[1]/2- crop_size[1]/2)
    for filename in os.listdir(dir):
        train_images.append(cv2.resize(cv2.imread(dir+filename),size)[x:x+crop_size[0],y:y+crop_size[1]])
    return train_images

