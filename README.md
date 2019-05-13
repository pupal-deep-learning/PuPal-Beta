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

- Haar cascade source: https://github.com/opencv/opencv/tree/master/data/haarcascades

We are working in a new and better eye detection Haar Cascade. Once ready we will make it available.

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

Download the model (see link above). Clone the repo onto to your computer and place the model.h5 into the pupal-app folder. 
To run the app you have two options:
- live with a webcam
- using pre-recorded video (we are working on it... To come soon)

**Live Webcam**

To run the code live, using a webcam, go to pupal-app folder and type in your terminal:
```
python pupal-webcam.py
```

Your webcam devices will be listed and you will be asked to selected and introduce the device number you want to use.
Typically it will be 0 for built-in webcams, but if you have more than one webcam selected between 0 or 1.

It will also give you an option to save the results into a text file which you can use to analyse the results.

Selected the best distance and position from the webcam so that proper measurements can be recorded.

The webcam live video will be visible as well as the predictons and ratios being calculated. This will help you to select the best webcam position and distance and show you what the app is measuring.

![image_w_predictions](https://user-images.githubusercontent.com/47978862/57651181-ea86ed00-75cc-11e9-8ad2-c7da48e11bde.jpg)


**Pre-Recorded Video**

An option to analyse pre-recorded video is being prepared and will be available soon.


# 5. Recommendations

The app runs on both CPU and GPU, however the use of a GPU with CUDA is higly recommended.
If you have a GPU with CUDA make sure to have the proper card drivers and CUDA installed.

Also tensorflow-gpu will be necessary: https://www.tensorflow.org/install/gpu

If you are using linux (Ubunbtu 18.04 or above) you can check this tutorial to install and use tensorflow-gpu in a safe and simple way without the need to install CUDA:

https://www.pugetsystems.com/labs/hpc/Install-TensorFlow-with-GPU-Support-the-Easy-Way-on-Ubuntu-18-04-without-installing-CUDA-1170/

You may need to install some requirements again in the conda environment.

NB: if you have a problem running the App, please check the requirements.
