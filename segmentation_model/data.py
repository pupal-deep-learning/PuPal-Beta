from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans

# # preprocessing: reflection removal
# from scipy.ndimage.morphology import binary_dilation
# from scipy.ndimage import generate_binary_structure
# from skimage.filters.rank import bottomhat
# from skimage.morphology import white_tophat
# from skimage.filters import median, gaussian
# from skimage.morphology import disk
# preprocessing: image equalization
from skimage import exposure
#from imgaug import augmenters as iaa

## imgaug

Iris = [255,255,255]
Unlabelled = [0,0,0]
COLOR_DICT = np.array([Iris,Unlabelled])

def binarize(img, thresh):
    return (thresh < img).astype('float64')

def imgaug_augs(img):
    return img
#def imgaug_augs(img):
#    img = (img).astype("uint8")
#     
#    seq = iaa.Sequential([
#            iaa.AdditiveGaussianNoise(scale=(0.1*255)), # Medium = 0.1*255 / Hard Core = 0.2*255 / Extreme = 0.3*255
#            iaa.GammaContrast(gamma = 1), # Medium = 1 / Hard core = 1.5 / Extreme = 3
#            iaa.GaussianBlur(sigma=(0.5))     # Medium = 0.5 / Hard core = 2.5 / Extreme = 5
#    ]) 
    
#    img = seq.augment_image(img)
    
#    return img

def crop(img, crop_size):
    # Note: image_data_format is 'channel_last'
    #assert img.shape[2] == 3
    height,width = img.shape[0], img.shape[1]
    crop_height, crop_width = crop_size[0], crop_size[1]
    startx = width//2-(crop_width//2)
    starty = height//2-(crop_height//2)

    #if len(img.shape) == 3:
    #return img[starty:starty+crop_height,startx:startx+crop_width, :]
    #elif len(img.shape) == 2:
    return img[starty:starty+crop_height,startx:startx+crop_width]



def contrast(img):
    # Get brightness range - i.e. darkest and lightest pixels
    min=np.min(img)        # result=144
    max=np.max(img)        # result=216

    # Make a LUT (Look-Up Table) to translate image values
    LUT=np.zeros(256,dtype=np.int32)
    LUT[min:max+1]=np.linspace(start=0,stop=255,num=(max-min)+1,endpoint=True,dtype=np.int32)

    return LUT[img]



def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        mask = binarize(mask, 0.5)
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



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,img_preprocessing=imgaug_augs,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict) #,preprocessing_function=imgaug_augs)
    mask_datagen = ImageDataGenerator(**aug_dict)
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
    train_generator = zip(image_generator, mask_generator) # create a tuple
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        img,mask = crop(img, target_size), crop(mask, target_size)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True,
                  save_to_dir = None,image_save_prefix  = "image"):
    #for time, img in img_dict.items():#for i in range(num_image):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray) 
        
        ## image equalization
        #img = exposure.equalize_hist(img)
        #img = (img * 255).astype('int32')
        #img = (img / 255).astype("float64")
        
#        img = trans.resize(img,target_size)
        
#         ## interpolate glint
#         # tophat filter to maximise bright spots
#         top_img = img.copy()
#         top=white_tophat(img, disk(5))
#         top_img=(img+top)

#         # median filter to get interpolated values
#         med_img = median(img, disk(5))

#         # threshold
#         thresh_img = top_img > np.percentile(top_img, 97.5)
#         struct = generate_binary_structure(2, 5)
#         thresh_img = binary_dilation(thresh_img,structure=struct).astype(thresh_img.dtype)
        
#         # impute interpolated values
#         img[thresh_img] = med_img[thresh_img]
        
#         # bottomhat filter to maximise dark spots 
#         bottom_img = bottomhat(img, disk(5))
#         img = (img+bottom_img)
        
#         # contrast to spread midrange
#         img = contrast(img)
                      
#         # gaussian smoothing to remove artifacts
#         img = gaussian(img, sigma=1, preserve_range=True).astype("int32")
        
#         # convert back to float
#         img = (img / 255).astype("float64")
        
#         # crop
        img = crop(img, target_size)
        
        if save_to_dir is not None:
            #io.imsave(save_to_dir + image_save_prefix + str(i) + '.png', img)
            io.imsave(save_to_dir + image_save_prefix + "_" + time + '.png', img)

        # reshape and add batch dims
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
               
        yield img


        
def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255



def saveResult(save_path,npyfile,savename_list=None,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        #img[img > 0.1] = 1
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        #io.imsave(os.path.join(save_path,"%d_predict.png"%int(savename_list[i])),img)
