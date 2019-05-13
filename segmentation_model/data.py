from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import exposure

def binarize(img, thresh):
    return (thresh < img).astype('float64')

def imgaug_augs(img):
    return img


def crop(img, crop_size):

    height,width = img.shape[0], img.shape[1]
    crop_height, crop_width = crop_size[0], crop_size[1]
    startx = width//2-(crop_width//2)
    starty = height//2-(crop_height//2)

    return img[starty:starty+crop_height,startx:startx+crop_width]


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        img,mask = crop(img, target_size), crop(mask, target_size)

        yield (img,mask)



def testGenerator(test_dict, num_image = 30, target_size = (256,256), flag_multi_class = False, as_gray = True,
                  save_to_dir = None, image_save_prefix  = "image"):
    for time, img in test_dict.items():#for i in range(num_image):
        img = trans.resize(img,target_size)
        
        if save_to_dir is not None:
            io.imsave(save_to_dir + image_save_prefix + time + '.png', img)

        # reshape and add batch dims
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
               
        yield img


def testGenerator(test_path,num_image = 30, target_size = (256,256), flag_multi_class = False, as_gray = True,
                  save_to_dir = None, image_save_prefix  = "image"):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray) 

        img = crop(img, target_size)
        
        if save_to_dir is not None:
            #io.imsave(save_to_dir + image_save_prefix + str(i) + '.png', img)
            io.imsave(save_to_dir + image_save_prefix + "_" + time + '.png', img)

        # reshape and add batch dims
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
               
        yield img
