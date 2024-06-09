# The DeepDream Experiments 

This notebook is inspirated and associated with the [Google DeepDream algorithm](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html),[DeepDream | Tensorflow](https://www.tensorflow.org/tutorials/generative/deepdream?hl=zh-cn) and [pytorch-deepdream by Aleksa Gordić](https://github.com/gordicaleksa/pytorch-deepdream/blob/master/deepdream.py) !

<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Test_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu3_3_pyrsize_4_pyrratio_1.8_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" alt="DeepDream example" align="center" style="width: 500px;"/> <br/>

**A problem I haven't solved**

But I failed to use the tensorflow core, It always show that the 'keras' is mot defined, although I have check that the version of my tensorflow, make sure that the offical tensorflow contain the keras- gpu, and I have tried several version, it still does not work, so I back to pytorch to work on my project.


In this project I'll be focusing on CNN: the VGG16, the dataset is ImageNet.

## Table of Contents
* [What is DeepDream?](#what-is-deepdream)
* [Samples of Deep Dream pictures](#Samples-of-Deep-Dream-pictures)
* [Learning material](#learning-material)

### What is DeepDream?
DeepDream is a computer vision program created by Google engineer Alexander Mordvintsev that uses a convolutional neural network to find and enhance patterns in images via algorithmic pareidolia, thus creating a dream-like appearance reminiscent of a psychedelic experience in the deliberately overprocessed images.--[Wikipadia.DeepDream](https://en.wikipedia.org/wiki/DeepDream).

So from an input image like the one on the left after "dreaming" we get the image on the right:
<p align="center">
<img src="data/input/Test.jpg" width="400"/>
<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Test_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu5_3_pyrsize_1_pyrratio_1.8_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="400"/>
</p>

### Samples of Deep Dream pictures
#### 1.Impact of increasing the pyramid size
<p align="center">
<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Test_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu5_3_pyrsize_1_pyrratio_1.8_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="270"/>
<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Test_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu5_3_pyrsize_3_pyrratio_1.8_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="270"/>
<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Test_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu5_3_pyrsize_4_pyrratio_1.8_iter_20_lr_0.09_shift_32_smooth_0.5.jpg" width="270"/>
</p>

Going from left to right the only parameter that changed was the pyramid size (from left to right: 1, 3, 4 levels).

#### 2.Impact of layer of use
<p align="center">
<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Test_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu5_1_pyrsize_4_pyrratio_1.8_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="270"/>
<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Test_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu3_3_pyrsize_4_pyrratio_1.8_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="270"/>
<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Test_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu4_2_pyrsize_4_pyrratio_1.8_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="270"/>
</p>

Going from left to right the only parameter that changed was the layer (from left to right: relu5_1, relu3_3, relu4_2).

#### 3.Impact of iterations
<p align="center">
<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Cloud_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu5_3_pyrsize_4_pyrratio_1.8_iter_2_lr_0.09_shift_32_smooth_0.5.jpg" width="270"/>
<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Cloud_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu5_3_pyrsize_4_pyrratio_1.8_iter_5_lr_0.09_shift_32_smooth_0.5.jpg" width="270"/>
<img src="data/out-images/VGG16_EXPERIMENTAL_IMAGENET/Cloud_width_600_model_VGG16_EXPERIMENTAL_IMAGENET_relu5_3_pyrsize_5_pyrratio_1.8_iter_10_lr_0.09_shift_32_smooth_0.5.jpg" width="270"/>
</p>

Going from left to right the only parameter that changed was the number of iterations (from left to right: 2, 5, 10).

### learning material
