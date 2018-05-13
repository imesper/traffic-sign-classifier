# **Traffic Sign Recognition**

## A Convolutional Deep Neural Network for Traffic Sign Classification

### Being able to correctly identify traffic signs is a crucial task for autonomous driving. It is a common and rich source of information that forms the base of important decision making for the autonomous system.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # 'Image References'
[hist_aug]: ./images/Hist_augmented.png 'Histogram of Augmented Data'
[hist_data]: ./images/Hist_train.png 'Histogram of Original Data'
[hist_val]: ./images/Hist_valid.png 'Histogram of Validation Data'
[hist_test]: ./images/Hist_test.png 'Histogram of Test Data'
[image1]: ./images/Normal_ex1.png 'Visualization'
[image_gray1]: ./images/Gray_ex1.png 'Grayscaling'
[image_clahe1]: ./images/CLAHE_ex1.png
[image_rotated1]: ./images/Rotated_ex1.png 'Random Noise'
[image4]: ./Web/arrow-points-up.jpg 'Traffic Sign 1'
[image5]: ./Web/caution.jpg 'Traffic Sign 2'
[image6]: ./Web/70kmh.jpg 'Traffic Sign 3'

[image7]: ./Web/pedestrians.jpg'Traffic Sign 4'
[image8]: ./Web/Road_Work.jpg 'Traffic Sign 5'

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation. Here is a link to my code [project code](https://github.com/imesper/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

## Data Set Summary & Exploration

The python list len() method was used to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### Histogram of Data Set.

Here is an exploratory visualization of the data set.

This histogram show the original data. On the x-axis are the classes of the signs and on the y-axis the number os samples.

![Histogram of train data][hist_data]

## Design and Test a Model Architecture

### Augmenting the Data

As observed on the histagram some classes has much less samples than others. In other to address this shortcome we generated data based on the inverse probability of the amount of data. So, if a class has less samples than other it will much more likely to be augmented.
The function below shows the code used to accomplish this.
An multiplication by 2 on max_value is used to half down the probabilities. Using this approach still guarentees that even classes with large amount of samples are augmented.

```python
def normInverseDictProbability(data):
    value_sum = sum(data.values())
    max_value = max(data.values()) * 2
    norm_data = {}
    for i in range(len(data)):
        norm_data[i] = (max_value - data[i]) / max_value
    return norm_data
```

To generate new data, we used two methods, first was to rotate the images randomly from -20 degrees to 20 degrees, as shown in the code below.

```python
def generateRotatedImage(image):
    degree = random.randrange(-20, 20)
    M = cv2.getRotationMatrix2D(
        (image_shape[0]/2, image_shape[1]/2), degree, 1)
    dst = cv2.warpAffine(image, M, (image_shape[0], image_shape[1]), )
    return dst
```

The second methos was to use an affine transformation of the image. The transformation is well explained [here](https://docs.opencv.org/3.1.0/d4/d61/tutorial_warp_affine.html).
The points to do the transformation was randomly picked with the following constraints:

* Point 1 x: From pixel 0 to 10
* Point 1 y: From pixel 0 to 10
* Point 2 x: From pixel 20 to 30
* Point 2 y: From pixel 0 to 10
* Point 3 x: From pixel 0 to 10
* Point 4 y: From pixel 20 to 30

Doing this we prevent a distortion that could make the image not recognizable.

```python
def generateAffineTransformedImage(image):
    pt1 = random.randrange(0, 10)
    pt2 = random.randrange(20, 30)
    pt3 = random.randrange(0, 10)
    pt1_2 = random.randrange(0, 10)
    pt2_2 = random.randrange(0, 10)
    pt3_2 = random.randrange(20, 30)

    var = 3
    pt1_var = random.randrange(-var, var)
    pt2_var = random.randrange(-var, var)
    pt3_var = random.randrange(-var, var)

    pt1_2_var = random.randrange(-var, var)
    pt2_2_var = random.randrange(-var, var)
    pt3_2_var = random.randrange(-var, var)

    pts1 = np.float32([[pt1, pt1_2], [pt2, pt2_2], [pt3, pt3_2]])
    pts2 = np.float32([[pt1 + pt1_var, pt1_2 + pt1_2_var], [pt2 + pt2_var, pt2_2 + pt2_2_var], [pt3 + pt3_var, pt3_2+pt3_2_var]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(image, M, image_shape[0], image_shape[1]))
    return dst
```

Here is an example of an original image and an augmented rotated image:

![alt text][image1] ![alt text][image_rotated1]

After augmenting the data with 30000 new images the histogram was more even as shown in the figure on the right below:

![Histogram of train data][hist_data] ![Histogram Augmented][hist_aug]

## Pre-Processing the Images

As a first step, I decided to convert the images to grayscale because it makes the processing lighter and did not affected the accuracy, as cited on other related work.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1] ![alt text][image_gray1]

The seecond step was to apply a histogram normalization on the image. The theory is explained [here](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html). The method used was the clahe (Contrast Limited Adaptive Histogram Equalization) becaues it showed a better result compared to a simple histogram normalization.

Here is an example of a traffic sign image before and after clahe.

![alt text][image_gray1] ![alt text][image_clahe1]

As a last step, I normalized the image data because it

I decided to generate additional data because it helps on optimizations.

#### 2. CNN Architecture

My final model consisted of the following layers:

|      Layer      |                 Description                 |
| :-------------: | :-----------------------------------------: |
|      Input      |           32x32x1 Grayscale image           |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x32 |
|      RELU       |                                             |
|   Max pooling   |        2x2 stride, outputs 14x14x32         |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 10x10x64 |
|      RELU       |                                             |
|   Max pooling   |         2x2 stride, outputs 5x5x64          |
| Convolution 5x5 |  1x1 stride, same padding, outputs 5x5x128  |
|      RELU       |                                             |
|   Max pooling   |         2x2 stride, outputs 3x3x128         |
|     Flatten     |         Conv1+Conv2+Conv3 size=9024         |
| Fully connected |           Input 9024 Output 3000            |
| Fully connected |            Input 3000 Output 400            |
| Fully connected |            Input 400 Output 120             |
| Fully connected |             Input 120 Output 84             |
| Fully connected |             Input 84 Output 43              |
|     Softmax     |    Softmax_cross_entropy_with_logits_v2     |
|                 |                                             |
|                 |                                             |

All of the parameters were found on an empirical basis. Several runs on training was done, and observations led to these optimal values.

To train the model, I used an Adam Optimizer wuth batch size of 128 samples. We used 10 epochs on a learning rate of 0.001. These were the best parameters tested during the development of the network.

My final model results were:

* training set accuracy of 99.2%
* validation set accuracy of 98.2%
* test set accuracy of 96.8%

If an iterative approach was chosen:

The first architecture was LeNet, due to the fact that it is a good neural network for image classification.

The LeNet did not achieve the desired result, so several changes was made untill a satisfactory model delivered a good accuracy.

We increased the filters of the first two convolutional layers and add a third layer with 128 filters. Dropouts layers was added to prevent overfitting.

It was tried several variations of the actual model, with more filters and less filter, it was added more dropouts and no droupouts at all. On a regular basis, the others models tested had an accuracuracy between 92% and 95%.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![Pedestrians](/home/ian/Development/SelfDrivingCar/CarND-Traffic-Sign-Classifier-Project/Web/pedestrians.jpg){:height="250px" width="250px"} ![Caution](/home/ian/Development/SelfDrivingCar/CarND-Traffic-Sign-Classifier-Project/Web/caution.jpg){:height="250px" width="250px"} ![70](/home/ian/Development/SelfDrivingCar/CarND-Traffic-Sign-Classifier-Project/Web/70kmh.jpg){:height="250px" width="250px"} ![AheadOnly](/home/ian/Development/SelfDrivingCar/CarND-Traffic-Sign-Classifier-Project/Web/arrow-points-up.jpg){:height="250px" width="250px"} ![Road](/home/ian/Development/SelfDrivingCar/CarND-Traffic-Sign-Classifier-Project/Web/Road_Work.jpg){:height="250px" width="250px"}

The images varies in size, but all have the sign on most of the image.

Here are the results of the prediction:

|      Image      |   Prediction    |
| :-------------: | :-------------: |
|   Pedestrians   |   Pedestrians   |
| General caution | General caution |
|   Ahead Only    |   Ahead Only    |
|     70 km/h     |     70 km/h     |
|    Road Work    |    Road Work    |

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a speed limit (70km/h) . The top five soft max probabilities were

Speed limit (70km/h) [ 4 0 1 8 15][0.36755636 0.1672712 0.167132 0.15353769 0.14450277]
Pedestrians [27 26 24 22 28][0.30931866 0.2133015 0.19082403 0.14329961 0.14325623]
Road work [25 29 22 24 14][0.41135165 0.20969748 0.20800371 0.12864663 0.04230047]
General caution [18 26 27 32 22][0.3491316 0.2236749 0.17059113 0.1292377 0.12736468]

| Probability |      Prediction       |
| :---------: | :-------------------: |
|     .36     | Speed limit (70km/h)  |
|     .16     |      No vehicles      |
|     .16     | Speed limit (20km/h)  |
|     .15     |   Turn right ahead    |
|     .14     | Speed limit (120km/h) |

For the second image.

| Probability |              Prediction               |
| :---------: | :-----------------------------------: |
|     .30     |              Pedestrians              |
|     .21     |            Traffic signals            |
|     .19     | Right-of-way at the next intersection |
|     .14     |            General caution            |
|     .14     |           Children crossing           |

For the third image.

| Probability |        Prediction         |
| :---------: | :-----------------------: |
|     .41     |         Road work         |
|     .20     |     Bicycles crossing     |
|     .20     |        Bumpy road         |
|     .12     | Road narrows on the right |
|     .04     |           Stop            |

For the fourth image.

| Probability |             Prediction              |
| :---------: | :---------------------------------: |
|    . 34     |           General caution           |
|     .22     |           Traffic signals           |
|     .17     |             Pedestrians             |
|     .12     | End of all speed and passing limits |
|     .12     |             Bumpy road              |

Ahead only [35 19 3 33 34][0.49320123 0.14689738 0.13453919 0.1315758 0.09378643]
For the fifth image.

| Probability |         Prediction          |
| :---------: | :-------------------------: |
|    . 49     |         Ahead only          |
|     .14     | Dangerous curve to the left |
|     .13     |    Speed limit (60km/h)     |
|     .13     |      Turn right ahead       |
|     .09     |       Turn left ahead       |
