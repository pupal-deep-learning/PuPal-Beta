![pupal_logo](https://user-images.githubusercontent.com/30341704/57581247-13b45a00-74b5-11e9-9c7c-eb2a804ecd68.png)

**Objective:**

PuPal is a deep learning application for segmentation of the iris from visible light images typically from portable devices.

**Overview** 

The approach was divided in 3 main parts: 
1. Capture of the eyes by Haar Cascade from a webcam
2. Segmentation of the iris (which also underline the pupil)
3. Algorithm fitting circles around the iris and the pupil to measure pupil/iris ratio

![strategy](https://user-images.githubusercontent.com/30341704/57581277-71e13d00-74b5-11e9-8459-42e4af00b4a4.png)

**1. Eye capture**


**2. Iris segmentation with Unet**
Unet architecture was used to train a deep learning network. The resulting model transform the eye region into a segmented picture. 


Download model from this link: https://drive.google.com/file/d/1ynVTNG_9bVT8IwJ9GWCeT2DiwW9Q5zV4/view?usp=sharing

**3. Measure pupil/iris ratio**
We used the ratio to avoid the problem of distance between the eye and the webcam. Plus the ratio is a good indicator of pupil dilation/constriction.


