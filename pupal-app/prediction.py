from model import *
from data import *
import os
import skimage.transform as trans

model = unet()
model.load_weights('with_augm_1-04_model.h5')
target_size = (256,256)

def predict(img):
    # reshape and add batch dims
    img = trans.resize(img,target_size)
    img = np.reshape(img,img.shape+(1,))
    img = np.reshape(img,(1,)+img.shape)

    # prediction
    result = model.predict(img)

    return result
