![pupal_logo](https://user-images.githubusercontent.com/30341704/57581247-13b45a00-74b5-11e9-9c7c-eb2a804ecd68.png)

**Objective:**

PuPal is a deep learning application for segmentation of the iris from visible light images typically from portable devices.

**Overview** 

The approach was divided in 3 main parts: 
1. Capture of the eyes by Haar Cascade from a webcam
2. Segmentation of the iris (which also underline the pupil)
3. Algorithm fitting circles around the iris and the pupil to measure pupil/iris ratio

![strategy](https://user-images.githubusercontent.com/30341704/57581277-71e13d00-74b5-11e9-8459-42e4af00b4a4.png)

# 1. Eye capture

![eye_capture](https://user-images.githubusercontent.com/30341704/57581349-7e19ca00-74b6-11e9-871a-b2e312ef71ee.png)



# 2. Iris segmentation with Unet

Unet architecture was used to train a deep learning network. The resulting model transforms the eye region into a segmented picture. 

![segmentation](https://user-images.githubusercontent.com/30341704/57581351-88d45f00-74b6-11e9-87e3-5f45c3c022c8.png)

- Architecture: Unet (described here: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- Loss: Jaccard index (also called Intersection over Union, https://en.wikipedia.org/wiki/Jaccard_index)
- Metrics: Jaccard index and Structural Similarity SSIM (https://en.wikipedia.org/wiki/Structural_similarity)

Download model from this link: https://drive.google.com/file/d/1ynVTNG_9bVT8IwJ9GWCeT2DiwW9Q5zV4/view?usp=sharing

# 3. Measure pupil/iris ratio

We used the ratio to avoid the problem of distance between the eye and the webcam. Plus the ratio is a good indicator of pupil dilation/constriction.

![circler](https://user-images.githubusercontent.com/30341704/57581372-e2d52480-74b6-11e9-9dcb-c21d01eb6078.png)

# 4. Run the application

Clone the repo, then in your Terminal you type:

```
source activate tf-gpu
```
For now it works with GPU, but we will try to make it works with CPU in the future.

Then to start the session:
```
python videoedit.py --o test
```

NB: if you have a problem running the App, please check the requirements.
