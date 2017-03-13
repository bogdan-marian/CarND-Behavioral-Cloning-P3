# **Behavioral Cloning**

## Writeup Template


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[center1]: ./examples/center_2017_03_13_00_39_55_074.jpg "center 1"
[recovery1]: ./examples/center_2017_03_13_00_15_39_476.jpg "recovery 1"
[recovery2]: ./examples/center_2017_03_13_00_15_40_075.jpg "recovery 2"
[recovery3]: ./examples/center_2017_03_13_00_15_41_193.jpg "recovery 3"
[recovery4]: ./examples/center_2017_03_13_00_15_41_792.jpg "recovery 4"
[recovery5]: ./examples/center_2017_03_13_00_15_42_368.jpg "recovery 5"
[mean_square]: ./examples/training_3_epochs_small.png "mean square 3 epochs"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* video.mp4 A video recording of my vehicle driving autonomously at least one lap around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model it is based on the NVIDIA neural network model presented in [deep-learning-self-driving-cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) article with added normalization and dropout layers at the input end.

The model itself consists of
* normalization layers (lines 56, 57)
* dropout layer (line 58)
* 3 convolution layers with 2x2 subsample and relu activation
* 2 convolution layers
* 1 flatten leayer (line 65)
* 4 fully connected layers (lines 66 to 69)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (line 58).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 15). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track ( [video.mp4](./video.mp4) ).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 71).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Also driving the track in the opposite direction helped allot.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the suggested NVIDIA network. I thought this model might be appropriate because it was pacifically designed for self driving cars and tested with very promising result on real cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, First I collected more training data by driving backwards and then I have added a dropout layer.

Then I continued training and testing on the simulator and when things went not as expected I was collecting recovery maneuvers and preventive maneuvers.

There were a few spots where the vehicle fell off the track. Special around tight corners. To improve the driving behavior in these cases, I created a local repository for me and recored recovery maneuvers and preventive maneuvers. If the added training set was improving the outcome then I will continue the process and if not then I was dropping the added training data and collected new one. I discovered in this process that the most affective data was the preventive maneuvers data. (not how to recover from the side of the road but what to do when the care was getting to close to the side of the road.)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 55-59) consisted of a convolution neural network with the following layers
* normalization layers (lines 56, 57)
* dropout layer (line 58)
* 3 convolution layers with 2x2 subsample and relu activation
* 2 convolution layers
* 1 flatten leayer (line 65)
* 4 fully connected layers (lines 66 to 69)

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps forword and one lap backwards on track lane driving. Here is an example image of center lane driving:

![alt text][center1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to do this by itself. These images show what a recovery looks like:

![alt text][recovery1]
![alt text][recovery2]
![alt text][recovery3]
![alt text][recovery4]
![alt text][recovery5]

After the collection process, I had 3427 number of data points (162.9 MB). I then preprocessed this data by dividing the color code by 255.0 and centering around 0 (line 56). I have also kept the cropping values suggested by the instructors. (line 57)


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the the fact that around this value the training_loss and validation_loss trend start diverging.

![alt text][mean_square]

Allso this was very obvious to me in observing the car driving using a model trained for more then 4 epochs (I have experimented with 4, 6, 15, 20). I never never felt as comfortable as with 3 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
I have to mention that git was a the most effective tool that helped me select good training data. I created a local repository collected more data and if that extra data actually ended up making the model worse then all I had to do was drop the new data using git and collect some more data using the new added value from the previous collection attempt and the training data would not get polluted by the previous bad collection session.   
