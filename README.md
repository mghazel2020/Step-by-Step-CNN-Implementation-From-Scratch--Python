# Step-by-Step Implementation of Convolutional Neural Networks (CNN) from Scratch to Classify MINIST Dataset in Python

<img src="images/CNN.jpeg" width="1000"/>

## 1. Objective

The objective of this project is to demonstrate the step-by-step implementation of a Convolutional Neural Network (CNN) from scratch to classify images of hand-written digits (0-9), using the MNIST data set.


## 2. Motivation

It is often said that “What I cannot build, I do not understand”. Thus, in order to gain a deeper understanding of Convolutional Neural Networks (CNN), I embarked on this project with the aim to built a CNN, step-by-step and from scratch in NumPy, without making use of any Deep Learning frameworks such as Tensorflow, Keras, etc. 

* Implementing a CNN from scratch is complex process, but it can be broken down to two phases

1. A forward phase, where the input is passed completely through the network:
  * During the forward phase, each layer will cache any data (like inputs, intermediate values, etc) it’ll need for the backward phase. This means that any backward phase must be preceded by a corresponding forward phase.

2. A backward phase, where gradients are backpropagated (backprop) and weights are updated:
  * During the backward phase, each layer will receive a gradient and also return a gradient. It will receive the gradient of loss with respect to its outputs and return the gradient of loss with respect to its inputs.

We shall illustrate the step-by-step implementation of these two main phases and break them down into several functionalities. We shall use the the MNIST handwritten standard dataset as the basis for learning and practicing how to develop, evaluate, and use convolutional neural networks for image classification from scratch. 

The structure of the implemented CNN is illustrated in the figure below. 

<img src="images/Implemented-CNN-Model.PNG" width="1000"/>

## 3. Data

* The MNIST database of handwritten digits, is widely used for training and evaluating various supervised machine and deep learning models [1]:

  * It has a training set of 60,000 examples
  * It has test set of 10,000 examples
  * It is a subset of a larger set available from NIST. 
  * The digits have been size-normalized and centered in a fixed-size image.
  * It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.
  * The original black and white images from NIST were size normalized and resized to 28x28 binary images.
  * Sample images from the MNIST data set are illustrated next:
    * There are significant variations how digits are handwritten by different people
    * The same digit may be written quite differently by different people
    * More significantly, different handwritten digits may appear similar, such as the 0, 5 and 6 or the 7 and 9.

<img src="images/MNIST-sample-images-02.webp" width="1000"/>


## 4. Development

In this section, we shall demonstrate how to develop a Convolutional Neural Network (CNN) for handwritten digit classification from scratch, without making use of any Deep Learning frameworks such as Tensorflow, Keras, etc. 

* The development process involves:

  * Reading and pre-processing the training and test data
  * Exploring and visualizing the training and test data:
    * Building the forward phase and backward phase of the CNN model
    * Training the built CNN model
    * Evaluating the performance of the trained CNN model.

* Author: Mohsen Ghazel (mghazel)
* Date: May 15th, 2021
* Project: Step-by-Step Implementation Convolutional Neural Networks (CNN) from Scratch using Numpy.

* Development Process:

1. Read and pre-process the training and test data:
  * We begin by reading the training and test MINIST datasets

2. Explore and visualize the training and test data:
  * We then inspect and visualize the training and test data sets

3. Build the CNN:
  * We then build the convolutional neural network, which consists of two phases:
  * A forward phase, where the input is passed completely through the network.
  * A backward phase, where gradients are backpropagated (backprop) and weights are updated.

4. Train the CNN:
  * We train the developed CNN on the training dataset.

5 Evaluate the performance of the trained CNN:
  * We shall evaluate the trained CNN based on the following performance evaluation metrics:
  * Training and validation loss as a function of the number of iterations
  * Training and validation accuracy as a function of the number of iterations
  * The confusion matrix
  * Examine sample mis-classified to get insights into their mis-classification.

### 4.1. Part 1: Imports and global variables:

#### 4.1.1. Standard scientific Python imports:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5; "># numpy
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#595979; "># tensorflow</span>
<span style="color:#200080; font-weight:bold; ">import</span> tensorflow <span style="color:#200080; font-weight:bold; ">as</span> tf
<span style="color:#595979; "># matplotlib</span>
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt
<span style="color:#595979; "># - import sklearn to use the confusion matrix function</span>
<span style="color:#200080; font-weight:bold; ">from</span> sklearn<span style="color:#308080; ">.</span>metrics <span style="color:#200080; font-weight:bold; ">import</span> confusion_matrix
<span style="color:#595979; "># import itertools</span>
<span style="color:#200080; font-weight:bold; ">import</span> itertools

<span style="color:#595979; "># random number generators values</span>
<span style="color:#595979; "># seed for reproducing the random number generation</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> seed
<span style="color:#595979; "># random integers: I(0,M)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> randint
<span style="color:#595979; "># random standard unform: U(0,1)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> random
<span style="color:#595979; "># time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime
<span style="color:#595979; "># I/O</span>
<span style="color:#200080; font-weight:bold; ">import</span> os
<span style="color:#595979; "># sys</span>
<span style="color:#200080; font-weight:bold; ">import</span> sys

<span style="color:#595979; "># library for savinga nd reading the model paramaters</span>
<span style="color:#200080; font-weight:bold; ">import</span> pickle
<span style="color:#595979; "># zipping variables into paired tuples </span>
<span style="color:#200080; font-weight:bold; ">import</span> gzip

<span style="color:#595979; "># Image package</span>
<span style="color:#200080; font-weight:bold; ">from</span> IPython<span style="color:#308080; ">.</span>display <span style="color:#200080; font-weight:bold; ">import</span> Image
<span style="color:#595979; "># using HTML code</span>
<span style="color:#200080; font-weight:bold; ">from</span> IPython<span style="color:#308080; ">.</span>core<span style="color:#308080; ">.</span>display <span style="color:#200080; font-weight:bold; ">import</span> HTML 

<span style="color:#595979; "># A function that renders the figure in a notebook, instead </span>
<span style="color:#595979; "># of displaying a dump of the figure object). </span>
<span style="color:#44aadd; ">%</span>matplotlib inline

<span style="color:#595979; "># check for successful package imports and versions</span>
<span style="color:#595979; "># python</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Python version : {0} "</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>sys<span style="color:#308080; ">.</span>version<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># numpy</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Numpy version  : {0}"</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># tensorflow</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Tensorflow version  : {0}"</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>tf<span style="color:#308080; ">.</span>__version__<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>


Python version <span style="color:#308080; ">:</span> <span style="color:#008000; ">3.8</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">5</span> <span style="color:#308080; ">(</span>default<span style="color:#308080; ">,</span> Sep  <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">2020</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">21</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">29</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">0</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">8</span><span style="color:#308080; ">)</span> <span style="color:#308080; ">[</span>MSC v<span style="color:#308080; ">.</span><span style="color:#008c00; ">1916</span> <span style="color:#008c00; ">64</span> bit <span style="color:#308080; ">(</span>AMD64<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> 
Numpy version  <span style="color:#308080; ">:</span> <span style="color:#008000; ">1.19</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">2</span>
Tensorflow version  <span style="color:#308080; ">:</span> <span style="color:#008000; ">2.4</span><span style="color:#308080; ">.</span><span style="color:#008c00; ">1</span>
</pre>

#### 4.1.2. Global variables:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#-------------------------------------------------------------------------------</span>
<span style="color:#595979; "># Set the random state to 101</span>
<span style="color:#595979; ">#-------------------------------------------------------------------------------</span>
<span style="color:#595979; "># - This ensures repeatable results everytime you run the code. </span>
RANDOM_STATE <span style="color:#308080; ">=</span> <span style="color:#008c00; ">101</span>

<span style="color:#595979; ">#-------------------------------------------------------------------------------</span>
<span style="color:#595979; "># We set the Numpy pseudo-random generator at a fixed value:</span>
<span style="color:#595979; ">#-------------------------------------------------------------------------------</span>
<span style="color:#595979; "># - This ensures repeatable results everytime you run the code. </span>
np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>seed<span style="color:#308080; ">(</span>RANDOM_STATE<span style="color:#308080; ">)</span>

<span style="color:#595979; "># the number of visualized images</span>
NUM_VISUALIZED_IMAGES <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span>
</pre>

### 4.2. Part 2: Load and explore the MNIST Dataset

#### 4.2.1. Load the MNIST dataset :

* Load the MNIST dataset of handwritten digits:
  * 60,000 labelled training examples
  * 10,000 labelled test examples
  * Each handwritten example is 28x28 pixels binary image.


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Load in the data: MNIST</span>
mnist <span style="color:#808030; ">=</span> tf<span style="color:#808030; ">.</span>keras<span style="color:#808030; ">.</span>datasets<span style="color:#808030; ">.</span>mnist
<span style="color:#696969; "># mnist.load_data() automatically splits traing and test data sets</span>
<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">,</span> y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> <span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">,</span> y_test<span style="color:#808030; ">)</span> <span style="color:#808030; ">=</span> mnist<span style="color:#808030; ">.</span>load_data<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
</pre>


#### 4.2.2. Display the number and shape of the training and test subsets:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Training data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of training images</span>
num_train_images <span style="color:#808030; ">=</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Training data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_train.shape: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of training images: "</span><span style="color:#808030; ">,</span> num_train_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>

<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># Test data:</span>
<span style="color:#696969; ">#------------------------------------------------------</span>
<span style="color:#696969; "># the number of test images</span>
num_test_images <span style="color:#808030; ">=</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">]</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Test data:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"x_test.shape: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Number of test images: "</span><span style="color:#808030; ">,</span> num_test_images<span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Image size: "</span><span style="color:#808030; ">,</span> x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">[</span><span style="color:#008c00; ">1</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training data<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_train<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">60000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
Number of training images<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">60000</span>
Image size<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Test data<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
x_test<span style="color:#808030; ">.</span>shape<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">10000</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
Number of test images<span style="color:#808030; ">:</span>  <span style="color:#008c00; ">10000</span>
Image size<span style="color:#808030; ">:</span>  <span style="color:#808030; ">(</span><span style="color:#008c00; ">28</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">28</span><span style="color:#808030; ">)</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

#### 4.2.3. Display the targets/classes:

* The classification of the digits should be: 0 to 9


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"Classes/labels:"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">'The target labels: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>np<span style="color:#808030; ">.</span>unique<span style="color:#808030; ">(</span>y_train<span style="color:#808030; ">)</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
<span style="color:#800000; font-weight:bold; ">print</span><span style="color:#808030; ">(</span><span style="color:#0000e6; ">"----------------------------------------------"</span><span style="color:#808030; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Classes<span style="color:#44aadd; ">/</span>labels<span style="color:#808030; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The target labels<span style="color:#808030; ">:</span> <span style="color:#808030; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#008c00; ">1</span> <span style="color:#008c00; ">2</span> <span style="color:#008c00; ">3</span> <span style="color:#008c00; ">4</span> <span style="color:#008c00; ">5</span> <span style="color:#008c00; ">6</span> <span style="color:#008c00; ">7</span> <span style="color:#008c00; ">8</span> <span style="color:#008c00; ">9</span><span style="color:#808030; ">]</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

#### 4.2.4. Examine the number of images for each class of the training and testing subsets:

##### 4.2.4.1. First implement a functionality to generate the histogram of the number of training and test images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># create a histogram of the number of images in each class/digit:</span>
<span style="color:#200080; font-weight:bold; ">def</span> plot_bar<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">,</span> relative<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    width <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.35</span>
    <span style="color:#200080; font-weight:bold; ">if</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">0.5</span>
    <span style="color:#200080; font-weight:bold; ">elif</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'right'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.5</span>
     
    <span style="color:#595979; "># calculate counts per type and sort, to ensure their order</span>
    unique<span style="color:#308080; ">,</span> counts <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> return_counts<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>
    sorted_index <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>argsort<span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span>
    unique <span style="color:#308080; ">=</span> unique<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
     
    <span style="color:#200080; font-weight:bold; ">if</span> relative<span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot as a percentage</span>
        counts <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">*</span>counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span><span style="color:#44aadd; ">/</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y<span style="color:#308080; ">)</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'% Count'</span>
    <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot counts</span>
        counts <span style="color:#308080; ">=</span> counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'Count'</span>
         
    xtemp <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>arange<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>bar<span style="color:#308080; ">(</span>xtemp <span style="color:#44aadd; ">+</span> n<span style="color:#44aadd; ">*</span>width<span style="color:#308080; ">,</span> counts<span style="color:#308080; ">,</span> align<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'center'</span><span style="color:#308080; ">,</span> alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">.7</span><span style="color:#308080; ">,</span> width<span style="color:#308080; ">=</span>width<span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span>xtemp<span style="color:#308080; ">,</span> unique<span style="color:#308080; ">,</span> rotation<span style="color:#308080; ">=</span><span style="color:#008c00; ">45</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Digit'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span>ylabel_text<span style="color:#308080; ">)</span>
<span style="color:#595979; "># add title</span>
plt<span style="color:#308080; ">.</span>suptitle<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Percentage of images per digit (0-9)'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

##### 4.2.4.2. Call the functionality to generate the histogram of the number of training and test images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># Call the function to create the histograms of the </span>
<span style="color:#595979; "># training and test images:</span>
<span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># set the figure size</span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">6</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># training data histogram</span>
plot_bar<span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># test data histogram</span>
plot_bar<span style="color:#308080; ">(</span>y_test<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'right'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># legend</span>
plt<span style="color:#308080; ">.</span>legend<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>
    <span style="color:#1060b6; ">'Training dataset: ({0} images)'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> 
    <span style="color:#1060b6; ">'Test dataset: ({0} images)'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_test<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> 
<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/train-test-data-histogram.png" width="1000"/>

#### 4.2.5. Visualize some of the training and test images and their associated targets:

##### 4.2.5.1. First implement a visualization functionality to visualize the number of randomly selected images:

<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; ">"""</span>
<span style="color:#696969; "># A utility function to visualize multiple images:</span>
<span style="color:#696969; ">"""</span>
<span style="color:#800000; font-weight:bold; ">def</span> visualize_images_and_labels<span style="color:#808030; ">(</span>num_visualized_images <span style="color:#808030; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#808030; ">,</span> dataset_flag <span style="color:#808030; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
  <span style="color:#696969; ">"""To visualize images.</span>
<span style="color:#696969; "></span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- num_visualized_images -- the number of visualized images (deafult 25)</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- dataset_flag -- 1: training dataset, 2: test dataset</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return:</span>
<span style="color:#696969; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- None</span>
<span style="color:#696969; ">&nbsp;&nbsp;"""</span>
  <span style="color:#696969; ">#--------------------------------------------</span>
  <span style="color:#696969; "># the suplot grid shape:</span>
  <span style="color:#696969; ">#--------------------------------------------</span>
  num_rows <span style="color:#808030; ">=</span> <span style="color:#008c00; ">5</span>
  <span style="color:#696969; "># the number of columns</span>
  num_cols <span style="color:#808030; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
  <span style="color:#696969; "># setup the subplots axes</span>
  fig<span style="color:#808030; ">,</span> axes <span style="color:#808030; ">=</span> plt<span style="color:#808030; ">.</span>subplots<span style="color:#808030; ">(</span>nrows<span style="color:#808030; ">=</span>num_rows<span style="color:#808030; ">,</span> ncols<span style="color:#808030; ">=</span>num_cols<span style="color:#808030; ">,</span> figsize<span style="color:#808030; ">=</span><span style="color:#808030; ">(</span><span style="color:#008c00; ">8</span><span style="color:#808030; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#808030; ">)</span><span style="color:#808030; ">)</span>
  <span style="color:#696969; "># set a seed random number generator for reproducible results</span>
  seed<span style="color:#808030; ">(</span>random_state_seed<span style="color:#808030; ">)</span>
  <span style="color:#696969; "># iterate over the sub-plots</span>
  <span style="color:#800000; font-weight:bold; ">for</span> row <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_rows<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
      <span style="color:#800000; font-weight:bold; ">for</span> col <span style="color:#800000; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#808030; ">(</span>num_cols<span style="color:#808030; ">)</span><span style="color:#808030; ">:</span>
        <span style="color:#696969; "># get the next figure axis</span>
        ax <span style="color:#808030; ">=</span> axes<span style="color:#808030; ">[</span>row<span style="color:#808030; ">,</span> col<span style="color:#808030; ">]</span><span style="color:#808030; ">;</span>
        <span style="color:#696969; "># turn-off subplot axis</span>
        ax<span style="color:#808030; ">.</span>set_axis_off<span style="color:#808030; ">(</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># if the dataset_flag = 1: Training data set</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">if</span> <span style="color:#808030; ">(</span> dataset_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#808030; ">)</span><span style="color:#808030; ">:</span> 
          <span style="color:#696969; "># generate a random image counter</span>
          counter <span style="color:#808030; ">=</span> randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>num_train_images<span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the training image</span>
          image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_train<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the target associated with the image</span>
          label <span style="color:#808030; ">=</span> y_train<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># dataset_flag = 2: Test data set</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#800000; font-weight:bold; ">else</span><span style="color:#808030; ">:</span> 
          <span style="color:#696969; "># generate a random image counter</span>
          counter <span style="color:#808030; ">=</span> randint<span style="color:#808030; ">(</span><span style="color:#008c00; ">0</span><span style="color:#808030; ">,</span>num_test_images<span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the test image</span>
          image <span style="color:#808030; ">=</span> np<span style="color:#808030; ">.</span>squeeze<span style="color:#808030; ">(</span>x_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">,</span><span style="color:#808030; ">:</span><span style="color:#808030; ">]</span><span style="color:#808030; ">)</span>
          <span style="color:#696969; "># get the target associated with the image</span>
          label <span style="color:#808030; ">=</span> y_test<span style="color:#808030; ">[</span>counter<span style="color:#808030; ">]</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        <span style="color:#696969; "># display the image</span>
        <span style="color:#696969; ">#--------------------------------------------</span>
        ax<span style="color:#808030; ">.</span>imshow<span style="color:#808030; ">(</span>image<span style="color:#808030; ">,</span> cmap<span style="color:#808030; ">=</span>plt<span style="color:#808030; ">.</span>cm<span style="color:#808030; ">.</span>gray_r<span style="color:#808030; ">,</span> interpolation<span style="color:#808030; ">=</span><span style="color:#0000e6; ">'nearest'</span><span style="color:#808030; ">)</span>
        <span style="color:#696969; "># set the title showing the image label</span>
        ax<span style="color:#808030; ">.</span>set_title<span style="color:#808030; ">(</span><span style="color:#0000e6; ">'y ='</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#808030; ">(</span>label<span style="color:#808030; ">)</span><span style="color:#808030; ">,</span> size <span style="color:#808030; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#808030; ">)</span>
</pre>

##### 4.2.5.2. Visualize some of the training images and their associated targets:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># call the function to visualize the training images</span>
visualize_images_and_labels<span style="color:#308080; ">(</span>NUM_VISUALIZED_IMAGES<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/train--sample-25-images.png" width="1000"/>

##### 4.2.5.3. Visualize some of the test images and their associated targets:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># call the function to visualize the training images</span>
visualize_images_and_labels<span style="color:#308080; ">(</span>NUM_VISUALIZED_IMAGES<span style="color:#308080; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">)</span>
</pre>

<img src="images/test--sample-25-images.png" width="1000"/>

4.2.6. Normalize the training and test images to the interval: [0, 1]:


<pre style="color:#000000;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#696969; "># Normalize the training images</span>
x_train <span style="color:#808030; ">=</span> x_train <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span>
<span style="color:#696969; "># Normalize the test images</span>
x_test <span style="color:#808030; ">=</span> x_test <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.0</span>
</pre>

### 4.3. Step 3: Build the Convolutional Network from Scratch:

* Building the convolutional neural network, which consists of two phases:

  * A forward phase, where the input is passed completely through the network.
  * A backward phase, where gradients are backpropagated (backprop) and weights are updated.

In this section, we shall implement the various functionalities of each of these phases for the following CNN model architecture.

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#-------------------------------------------------------------------------------</span>
<span style="color:#595979; "># Display the architecture of the implmented model:</span>
<span style="color:#595979; ">#-------------------------------------------------------------------------------</span>
<span style="color:#595979; "># height</span>
img_height <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1000</span>
<span style="color:#595979; "># width</span>
img_width <span style="color:#308080; ">=</span> <span style="color:#008c00; ">800</span>
<span style="color:#595979; "># display the image</span>
Image<span style="color:#308080; ">(</span>filename <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"Implemented-CNN-Model.PNG"</span><span style="color:#308080; ">,</span> width<span style="color:#308080; ">=</span>img_width<span style="color:#308080; ">,</span> height<span style="color:#308080; ">=</span>img_height<span style="color:#308080; ">)</span>
</pre>



<img src="images/Implemented-CNN-Model.PNG" width="1000"/>


#### 4.3.1. Forward Phase:

* The forward phase, where the input is passed completely through the network.

##### 4.3.1.1. Utilities functions:

* Utility methods for a Convolutional Neural Network:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> initializeFilter<span style="color:#308080; ">(</span>size<span style="color:#308080; ">,</span> scale <span style="color:#308080; ">=</span> <span style="color:#008000; ">1.0</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Utility method for initialzing the convolutional layers filters</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    stddev <span style="color:#308080; ">=</span> scale<span style="color:#44aadd; ">/</span>np<span style="color:#308080; ">.</span>sqrt<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>prod<span style="color:#308080; ">(</span>size<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">return</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>normal<span style="color:#308080; ">(</span>loc <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> scale <span style="color:#308080; ">=</span> stddev<span style="color:#308080; ">,</span> size <span style="color:#308080; ">=</span> size<span style="color:#308080; ">)</span>

<span style="color:#200080; font-weight:bold; ">def</span> initializeWeight<span style="color:#308080; ">(</span>size<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Utility method for initialzing the full-connected layers weights and biases</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    <span style="color:#200080; font-weight:bold; ">return</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>standard_normal<span style="color:#308080; ">(</span>size<span style="color:#308080; ">=</span>size<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">*</span> <span style="color:#008000; ">0.01</span>

<span style="color:#200080; font-weight:bold; ">def</span> nanargmax<span style="color:#308080; ">(</span>arr<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Utility method for computing the non-NAN argmax()</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    idx <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>nanargmax<span style="color:#308080; ">(</span>arr<span style="color:#308080; ">)</span>
    idxs <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>unravel_index<span style="color:#308080; ">(</span>idx<span style="color:#308080; ">,</span> arr<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">return</span> idxs   
</pre>

##### 4.3.1.2. Forward operations for a convolutional neural network:

* Functions for implementing the forward operations for the CNN:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> convolution<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> filt<span style="color:#308080; ">,</span> bias<span style="color:#308080; ">,</span> s<span style="color:#308080; ">=</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">'''</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Confolves `filt` over `image` using stride `s`</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;'''</span>
    <span style="color:#308080; ">(</span>n_f<span style="color:#308080; ">,</span> n_c_f<span style="color:#308080; ">,</span> f<span style="color:#308080; ">,</span> _<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> filt<span style="color:#308080; ">.</span>shape <span style="color:#595979; "># filter dimensions</span>
    n_c<span style="color:#308080; ">,</span> in_dim<span style="color:#308080; ">,</span> _ <span style="color:#308080; ">=</span> image<span style="color:#308080; ">.</span>shape <span style="color:#595979; "># image dimensions</span>
    
    out_dim <span style="color:#308080; ">=</span> <span style="color:#400000; ">int</span><span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>in_dim <span style="color:#44aadd; ">-</span> f<span style="color:#308080; ">)</span><span style="color:#44aadd; ">/</span>s<span style="color:#308080; ">)</span><span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span> <span style="color:#595979; "># calculate output dimensions</span>
    
    <span style="color:#200080; font-weight:bold; ">assert</span> n_c <span style="color:#44aadd; ">==</span> n_c_f<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">"Dimensions of filter must match dimensions of input image"</span>
    
    out <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>n_f<span style="color:#308080; ">,</span>out_dim<span style="color:#308080; ">,</span>out_dim<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># convolve the filter over every part of the image, adding the bias at each step. </span>
    <span style="color:#200080; font-weight:bold; ">for</span> curr_f <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>n_f<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        curr_y <span style="color:#308080; ">=</span> out_y <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
        <span style="color:#200080; font-weight:bold; ">while</span> curr_y <span style="color:#44aadd; ">+</span> f <span style="color:#44aadd; ">&lt;=</span> in_dim<span style="color:#308080; ">:</span>
            curr_x <span style="color:#308080; ">=</span> out_x <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
            <span style="color:#200080; font-weight:bold; ">while</span> curr_x <span style="color:#44aadd; ">+</span> f <span style="color:#44aadd; ">&lt;=</span> in_dim<span style="color:#308080; ">:</span>
                out<span style="color:#308080; ">[</span>curr_f<span style="color:#308080; ">,</span> out_y<span style="color:#308080; ">,</span> out_x<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span><span style="color:#400000; ">sum</span><span style="color:#308080; ">(</span>filt<span style="color:#308080; ">[</span>curr_f<span style="color:#308080; ">]</span> <span style="color:#44aadd; ">*</span> image<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span>curr_y<span style="color:#308080; ">:</span>curr_y<span style="color:#44aadd; ">+</span>f<span style="color:#308080; ">,</span> curr_x<span style="color:#308080; ">:</span>curr_x<span style="color:#44aadd; ">+</span>f<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> bias<span style="color:#308080; ">[</span>curr_f<span style="color:#308080; ">]</span>
                curr_x <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> s
                out_x <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
            curr_y <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> s
            out_y <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
        
    <span style="color:#200080; font-weight:bold; ">return</span> out

<span style="color:#200080; font-weight:bold; ">def</span> maxpool<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> f<span style="color:#308080; ">=</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span> s<span style="color:#308080; ">=</span><span style="color:#008c00; ">2</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">'''</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Downsample `image` using kernel size `f` and stride `s`</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;'''</span>
    n_c<span style="color:#308080; ">,</span> h_prev<span style="color:#308080; ">,</span> w_prev <span style="color:#308080; ">=</span> image<span style="color:#308080; ">.</span>shape
    
    h <span style="color:#308080; ">=</span> <span style="color:#400000; ">int</span><span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>h_prev <span style="color:#44aadd; ">-</span> f<span style="color:#308080; ">)</span><span style="color:#44aadd; ">/</span>s<span style="color:#308080; ">)</span><span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span>
    w <span style="color:#308080; ">=</span> <span style="color:#400000; ">int</span><span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>w_prev <span style="color:#44aadd; ">-</span> f<span style="color:#308080; ">)</span><span style="color:#44aadd; ">/</span>s<span style="color:#308080; ">)</span><span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span>
    
    downsampled <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>n_c<span style="color:#308080; ">,</span> h<span style="color:#308080; ">,</span> w<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">for</span> i <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>n_c<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># slide maxpool window over each part of the image and assign the max value at each step to the output</span>
        curr_y <span style="color:#308080; ">=</span> out_y <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
        <span style="color:#200080; font-weight:bold; ">while</span> curr_y <span style="color:#44aadd; ">+</span> f <span style="color:#44aadd; ">&lt;=</span> h_prev<span style="color:#308080; ">:</span>
            curr_x <span style="color:#308080; ">=</span> out_x <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
            <span style="color:#200080; font-weight:bold; ">while</span> curr_x <span style="color:#44aadd; ">+</span> f <span style="color:#44aadd; ">&lt;=</span> w_prev<span style="color:#308080; ">:</span>
                downsampled<span style="color:#308080; ">[</span>i<span style="color:#308080; ">,</span> out_y<span style="color:#308080; ">,</span> out_x<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span><span style="color:#400000; ">max</span><span style="color:#308080; ">(</span>image<span style="color:#308080; ">[</span>i<span style="color:#308080; ">,</span> curr_y<span style="color:#308080; ">:</span>curr_y<span style="color:#44aadd; ">+</span>f<span style="color:#308080; ">,</span> curr_x<span style="color:#308080; ">:</span>curr_x<span style="color:#44aadd; ">+</span>f<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
                curr_x <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> s
                out_x <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
            curr_y <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> s
            out_y <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
    <span style="color:#200080; font-weight:bold; ">return</span> downsampled

<span style="color:#200080; font-weight:bold; ">def</span> softmax<span style="color:#308080; ">(</span>X<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">'''</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Computes the Softmax()</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;'''</span>
    out <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>exp<span style="color:#308080; ">(</span>X<span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">return</span> out<span style="color:#44aadd; ">/</span>np<span style="color:#308080; ">.</span><span style="color:#400000; ">sum</span><span style="color:#308080; ">(</span>out<span style="color:#308080; ">)</span>

<span style="color:#200080; font-weight:bold; ">def</span> categoricalCrossEntropy<span style="color:#308080; ">(</span>probs<span style="color:#308080; ">,</span> label<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">'''</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Computes the categorical entropy</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;'''</span>
    <span style="color:#200080; font-weight:bold; ">return</span> <span style="color:#44aadd; ">-</span>np<span style="color:#308080; ">.</span><span style="color:#400000; ">sum</span><span style="color:#308080; ">(</span>label <span style="color:#44aadd; ">*</span> np<span style="color:#308080; ">.</span>log<span style="color:#308080; ">(</span>probs<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

<span style="color:#200080; font-weight:bold; ">def</span> predict<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> f1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> w3<span style="color:#308080; ">,</span> w4<span style="color:#308080; ">,</span> b1<span style="color:#308080; ">,</span> b2<span style="color:#308080; ">,</span> b3<span style="color:#308080; ">,</span> b4<span style="color:#308080; ">,</span> conv_s <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> pool_f <span style="color:#308080; ">=</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span> pool_s <span style="color:#308080; ">=</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">'''</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Make predictions with trained filters/weights. </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;'''</span>
    <span style="color:#595979; "># convolution operation</span>
    conv1 <span style="color:#308080; ">=</span> convolution<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> f1<span style="color:#308080; ">,</span> b1<span style="color:#308080; ">,</span> conv_s<span style="color:#308080; ">)</span> 
    <span style="color:#595979; ">#relu activation</span>
    conv1<span style="color:#308080; ">[</span>conv1<span style="color:#44aadd; ">&lt;=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span> 
    
    <span style="color:#595979; "># second convolution operation</span>
    conv2 <span style="color:#308080; ">=</span> convolution<span style="color:#308080; ">(</span>conv1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> b2<span style="color:#308080; ">,</span> conv_s<span style="color:#308080; ">)</span> 
    <span style="color:#595979; "># pass through ReLU non-linearity</span>
    conv2<span style="color:#308080; ">[</span>conv2<span style="color:#44aadd; ">&lt;=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span> 
    
    <span style="color:#595979; "># maxpooling operation</span>
    pooled <span style="color:#308080; ">=</span> maxpool<span style="color:#308080; ">(</span>conv2<span style="color:#308080; ">,</span> pool_f<span style="color:#308080; ">,</span> pool_s<span style="color:#308080; ">)</span> 
    <span style="color:#308080; ">(</span>nf2<span style="color:#308080; ">,</span> dim2<span style="color:#308080; ">,</span> _<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> pooled<span style="color:#308080; ">.</span>shape
    <span style="color:#595979; "># flatten pooled layer</span>
    fc <span style="color:#308080; ">=</span> pooled<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>nf2 <span style="color:#44aadd; ">*</span> dim2 <span style="color:#44aadd; ">*</span> dim2<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span> 
    
    <span style="color:#595979; "># first dense layer</span>
    z <span style="color:#308080; ">=</span> w3<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>fc<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> b3 
    <span style="color:#595979; "># pass through ReLU non-linearity</span>
    z<span style="color:#308080; ">[</span>z<span style="color:#44aadd; ">&lt;=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span> 
    
    <span style="color:#595979; "># second dense layer</span>
    out <span style="color:#308080; ">=</span> w4<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>z<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> b4 
    <span style="color:#595979; "># predict class probabilities with the softmax activation function</span>
    probs <span style="color:#308080; ">=</span> softmax<span style="color:#308080; ">(</span>out<span style="color:#308080; ">)</span> 
    
    <span style="color:#200080; font-weight:bold; ">return</span> np<span style="color:#308080; ">.</span>argmax<span style="color:#308080; ">(</span>probs<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span><span style="color:#400000; ">max</span><span style="color:#308080; ">(</span>probs<span style="color:#308080; ">)</span>
</pre>

##### 4.3.1.3. Backward operations for a convolutional neural network:

* Functions for implementing the backward operations for the CNN:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> convolutionBackward<span style="color:#308080; ">(</span>dconv_prev<span style="color:#308080; ">,</span> conv_in<span style="color:#308080; ">,</span> filt<span style="color:#308080; ">,</span> s<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">'''</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Backpropagation through a convolutional layer. </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;'''</span>
    <span style="color:#308080; ">(</span>n_f<span style="color:#308080; ">,</span> n_c<span style="color:#308080; ">,</span> f<span style="color:#308080; ">,</span> _<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> filt<span style="color:#308080; ">.</span>shape
    <span style="color:#308080; ">(</span>_<span style="color:#308080; ">,</span> orig_dim<span style="color:#308080; ">,</span> _<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> conv_in<span style="color:#308080; ">.</span>shape
    <span style="color:#595979; ">## initialize derivatives</span>
    dout <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span>conv_in<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span> 
    dfilt <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span>filt<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    dbias <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>n_f<span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">for</span> curr_f <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>n_f<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># loop through all filters</span>
        curr_y <span style="color:#308080; ">=</span> out_y <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
        <span style="color:#200080; font-weight:bold; ">while</span> curr_y <span style="color:#44aadd; ">+</span> f <span style="color:#44aadd; ">&lt;=</span> orig_dim<span style="color:#308080; ">:</span>
            curr_x <span style="color:#308080; ">=</span> out_x <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
            <span style="color:#200080; font-weight:bold; ">while</span> curr_x <span style="color:#44aadd; ">+</span> f <span style="color:#44aadd; ">&lt;=</span> orig_dim<span style="color:#308080; ">:</span>
                <span style="color:#595979; "># loss gradient of filter (used to update the filter)</span>
                dfilt<span style="color:#308080; ">[</span>curr_f<span style="color:#308080; ">]</span> <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> dconv_prev<span style="color:#308080; ">[</span>curr_f<span style="color:#308080; ">,</span> out_y<span style="color:#308080; ">,</span> out_x<span style="color:#308080; ">]</span> <span style="color:#44aadd; ">*</span> conv_in<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span> curr_y<span style="color:#308080; ">:</span>curr_y<span style="color:#44aadd; ">+</span>f<span style="color:#308080; ">,</span> curr_x<span style="color:#308080; ">:</span>curr_x<span style="color:#44aadd; ">+</span>f<span style="color:#308080; ">]</span>
                <span style="color:#595979; "># loss gradient of the input to the convolution operation (conv1 in the case of this network)</span>
                dout<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span> curr_y<span style="color:#308080; ">:</span>curr_y<span style="color:#44aadd; ">+</span>f<span style="color:#308080; ">,</span> curr_x<span style="color:#308080; ">:</span>curr_x<span style="color:#44aadd; ">+</span>f<span style="color:#308080; ">]</span> <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> dconv_prev<span style="color:#308080; ">[</span>curr_f<span style="color:#308080; ">,</span> out_y<span style="color:#308080; ">,</span> out_x<span style="color:#308080; ">]</span> <span style="color:#44aadd; ">*</span> filt<span style="color:#308080; ">[</span>curr_f<span style="color:#308080; ">]</span> 
                curr_x <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> s
                out_x <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
            curr_y <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> s
            out_y <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
        <span style="color:#595979; "># loss gradient of the bias</span>
        dbias<span style="color:#308080; ">[</span>curr_f<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span><span style="color:#400000; ">sum</span><span style="color:#308080; ">(</span>dconv_prev<span style="color:#308080; ">[</span>curr_f<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> dout<span style="color:#308080; ">,</span> dfilt<span style="color:#308080; ">,</span> dbias


<span style="color:#200080; font-weight:bold; ">def</span> maxpoolBackward<span style="color:#308080; ">(</span>dpool<span style="color:#308080; ">,</span> orig<span style="color:#308080; ">,</span> f<span style="color:#308080; ">,</span> s<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">'''</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;'''</span>
    <span style="color:#308080; ">(</span>n_c<span style="color:#308080; ">,</span> orig_dim<span style="color:#308080; ">,</span> _<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> orig<span style="color:#308080; ">.</span>shape
    
    dout <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span>orig<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">for</span> curr_c <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>n_c<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        curr_y <span style="color:#308080; ">=</span> out_y <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
        <span style="color:#200080; font-weight:bold; ">while</span> curr_y <span style="color:#44aadd; ">+</span> f <span style="color:#44aadd; ">&lt;=</span> orig_dim<span style="color:#308080; ">:</span>
            curr_x <span style="color:#308080; ">=</span> out_x <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
            <span style="color:#200080; font-weight:bold; ">while</span> curr_x <span style="color:#44aadd; ">+</span> f <span style="color:#44aadd; ">&lt;=</span> orig_dim<span style="color:#308080; ">:</span>
                <span style="color:#595979; "># obtain index of largest value in input for current window</span>
                <span style="color:#308080; ">(</span>a<span style="color:#308080; ">,</span> b<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> nanargmax<span style="color:#308080; ">(</span>orig<span style="color:#308080; ">[</span>curr_c<span style="color:#308080; ">,</span> curr_y<span style="color:#308080; ">:</span>curr_y<span style="color:#44aadd; ">+</span>f<span style="color:#308080; ">,</span> curr_x<span style="color:#308080; ">:</span>curr_x<span style="color:#44aadd; ">+</span>f<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
                dout<span style="color:#308080; ">[</span>curr_c<span style="color:#308080; ">,</span> curr_y<span style="color:#44aadd; ">+</span>a<span style="color:#308080; ">,</span> curr_x<span style="color:#44aadd; ">+</span>b<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> dpool<span style="color:#308080; ">[</span>curr_c<span style="color:#308080; ">,</span> out_y<span style="color:#308080; ">,</span> out_x<span style="color:#308080; ">]</span>
                
                curr_x <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> s
                out_x <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
            curr_y <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> s
            out_y <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
        
    <span style="color:#200080; font-weight:bold; ">return</span> dout
</pre>

##### 4.3.1.4. Full Forward-Backward pass for a convolutional neural network:

* Function for running a full forward-backward cycle for the CNN.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> conv_forward_backward_pass<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> label<span style="color:#308080; ">,</span> params<span style="color:#308080; ">,</span> conv_s<span style="color:#308080; ">,</span> pool_f<span style="color:#308080; ">,</span> pool_s<span style="color:#308080; ">,</span> lr<span style="color:#308080; ">=</span><span style="color:#008000; ">0.01</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;A function for running a full forward-backward cycle for the CNN:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    <span style="color:#595979; "># get the parameters</span>
    <span style="color:#308080; ">[</span>f1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> w3<span style="color:#308080; ">,</span> w4<span style="color:#308080; ">,</span> b1<span style="color:#308080; ">,</span> b2<span style="color:#308080; ">,</span> b3<span style="color:#308080; ">,</span> b4<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> params 
    
    <span style="color:#595979; ">#================================================</span>
    <span style="color:#595979; "># Step 1: Forward operation:</span>
    <span style="color:#595979; ">#================================================</span>
    <span style="color:#595979; "># Convolutional layer: Conv1:</span>
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># convolution operation</span>
    conv1 <span style="color:#308080; ">=</span> convolution<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> f1<span style="color:#308080; ">,</span> b1<span style="color:#308080; ">,</span> conv_s<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># pass through ReLU non-linearity</span>
    conv1<span style="color:#308080; ">[</span>conv1<span style="color:#44aadd; ">&lt;=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span> 
    
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># Convolutional layer: Conv2:</span>
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># second convolution operation</span>
    conv2 <span style="color:#308080; ">=</span> convolution<span style="color:#308080; ">(</span>conv1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> b2<span style="color:#308080; ">,</span> conv_s<span style="color:#308080; ">)</span> 
    <span style="color:#595979; "># pass through ReLU non-linearity</span>
    conv2<span style="color:#308080; ">[</span>conv2<span style="color:#44aadd; ">&lt;=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span> 
    
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># Maxpooling layer: pool:</span>
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># maxpooling operation</span>
    pooled <span style="color:#308080; ">=</span> maxpool<span style="color:#308080; ">(</span>conv2<span style="color:#308080; ">,</span> pool_f<span style="color:#308080; ">,</span> pool_s<span style="color:#308080; ">)</span> 
    <span style="color:#595979; "># get the dimensions</span>
    <span style="color:#308080; ">(</span>nf2<span style="color:#308080; ">,</span> dim2<span style="color:#308080; ">,</span> _<span style="color:#308080; ">)</span> <span style="color:#308080; ">=</span> pooled<span style="color:#308080; ">.</span>shape
    <span style="color:#595979; "># flatten pooled layer</span>
    fc <span style="color:#308080; ">=</span> pooled<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>nf2 <span style="color:#44aadd; ">*</span> dim2 <span style="color:#44aadd; ">*</span> dim2<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span> 
    
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># First dense layer</span>
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># Apply the linear function</span>
    z <span style="color:#308080; ">=</span> w3<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>fc<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> b3 
    <span style="color:#595979; "># pass through ReLU non-linearity</span>
    z<span style="color:#308080; ">[</span>z<span style="color:#44aadd; ">&lt;=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span> 
    
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># Second dense layer</span>
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># Apply the linear function</span>
    out <span style="color:#308080; ">=</span> w4<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>z<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> b4 
     
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># Softmax layer</span>
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># predict class probabilities with the softmax activation function</span>
    probs <span style="color:#308080; ">=</span> softmax<span style="color:#308080; ">(</span>out<span style="color:#308080; ">)</span>
    
    <span style="color:#595979; ">#================================================</span>
    <span style="color:#595979; "># Step 2: The loss function:</span>
    <span style="color:#595979; ">#================================================</span>
    <span style="color:#595979; "># compute the # categorical cross-entropy loss</span>
    loss <span style="color:#308080; ">=</span> categoricalCrossEntropy<span style="color:#308080; ">(</span>probs<span style="color:#308080; ">,</span> label<span style="color:#308080; ">)</span> 
        
    <span style="color:#595979; ">#================================================</span>
    <span style="color:#595979; "># Step 3: The Backward Operation</span>
    <span style="color:#595979; ">#================================================</span>
    <span style="color:#595979; "># derivative of loss w.r.t. final dense layer output</span>
    dout <span style="color:#308080; ">=</span> probs <span style="color:#44aadd; ">-</span> label 
    <span style="color:#595979; "># loss gradient of final dense layer weights</span>
    dw4 <span style="color:#308080; ">=</span> dout<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>z<span style="color:#308080; ">.</span>T<span style="color:#308080; ">)</span> 
    <span style="color:#595979; "># loss gradient of final dense layer biases</span>
    db4 <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span><span style="color:#400000; ">sum</span><span style="color:#308080; ">(</span>dout<span style="color:#308080; ">,</span> axis <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>b4<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span> 
    <span style="color:#595979; "># loss gradient of first dense layer outputs </span>
    dz <span style="color:#308080; ">=</span> w4<span style="color:#308080; ">.</span>T<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>dout<span style="color:#308080; ">)</span> 
    <span style="color:#595979; "># backpropagate through ReLU </span>
    dz<span style="color:#308080; ">[</span>z<span style="color:#44aadd; ">&lt;=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span> 
    dw3 <span style="color:#308080; ">=</span> dz<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>fc<span style="color:#308080; ">.</span>T<span style="color:#308080; ">)</span>
    db3 <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span><span style="color:#400000; ">sum</span><span style="color:#308080; ">(</span>dz<span style="color:#308080; ">,</span> axis <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>b3<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># loss gradients of fully-connected layer (pooling layer)</span>
    dfc <span style="color:#308080; ">=</span> w3<span style="color:#308080; ">.</span>T<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>dz<span style="color:#308080; ">)</span> 
    <span style="color:#595979; "># reshape fully connected into dimensions of pooling layer</span>
    dpool <span style="color:#308080; ">=</span> dfc<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>pooled<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span> 
    
    <span style="color:#595979; "># backprop through the max-pooling layer(only neurons with highest activation in window get updated)</span>
    dconv2 <span style="color:#308080; ">=</span> maxpoolBackward<span style="color:#308080; ">(</span>dpool<span style="color:#308080; ">,</span> conv2<span style="color:#308080; ">,</span> pool_f<span style="color:#308080; ">,</span> pool_s<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># backpropagate through ReLU</span>
    dconv2<span style="color:#308080; ">[</span>conv2<span style="color:#44aadd; ">&lt;=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span> 
    
    <span style="color:#595979; "># backpropagate previous gradient through second convolutional layer.</span>
    dconv1<span style="color:#308080; ">,</span> df2<span style="color:#308080; ">,</span> db2 <span style="color:#308080; ">=</span> convolutionBackward<span style="color:#308080; ">(</span>dconv2<span style="color:#308080; ">,</span> conv1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> conv_s<span style="color:#308080; ">)</span> 
    <span style="color:#595979; "># backpropagate through ReLU</span>
    dconv1<span style="color:#308080; ">[</span>conv1<span style="color:#44aadd; ">&lt;=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span> 
    
    <span style="color:#595979; "># backpropagate previous gradient through first convolutional layer.</span>
    dimage<span style="color:#308080; ">,</span> df1<span style="color:#308080; ">,</span> db1 <span style="color:#308080; ">=</span> convolutionBackward<span style="color:#308080; ">(</span>dconv1<span style="color:#308080; ">,</span> image<span style="color:#308080; ">,</span> f1<span style="color:#308080; ">,</span> conv_s<span style="color:#308080; ">)</span> 
    
    <span style="color:#595979; "># store the gradient of the parameters</span>
    grads <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span>df1<span style="color:#308080; ">,</span> df2<span style="color:#308080; ">,</span> dw3<span style="color:#308080; ">,</span> dw4<span style="color:#308080; ">,</span> db1<span style="color:#308080; ">,</span> db2<span style="color:#308080; ">,</span> db3<span style="color:#308080; ">,</span> db4<span style="color:#308080; ">]</span> 
    
    <span style="color:#595979; ">#================================================</span>
    <span style="color:#595979; "># Step 4: The Gradient-Descent Optimization:</span>
    <span style="color:#595979; ">#================================================</span>
    <span style="color:#595979; "># Apply Gradient Descent to update the paramaters:</span>
    <span style="color:#595979; ">#-----------------------------------------------</span>
    <span style="color:#595979; "># update f1:</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    f1 <span style="color:#44aadd; ">-</span><span style="color:#308080; ">=</span> lr <span style="color:#44aadd; ">*</span> df1
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># update f2:</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    f2 <span style="color:#44aadd; ">-</span><span style="color:#308080; ">=</span> lr <span style="color:#44aadd; ">*</span> df2
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># update w3:</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    w3 <span style="color:#44aadd; ">-</span><span style="color:#308080; ">=</span> lr <span style="color:#44aadd; ">*</span> dw3
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># update w4:</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    w4 <span style="color:#44aadd; ">-</span><span style="color:#308080; ">=</span> lr <span style="color:#44aadd; ">*</span> dw4
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># update b1:</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    b1 <span style="color:#44aadd; ">-</span><span style="color:#308080; ">=</span> lr <span style="color:#44aadd; ">*</span> db1
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># update b2:</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    b2 <span style="color:#44aadd; ">-</span><span style="color:#308080; ">=</span> lr <span style="color:#44aadd; ">*</span> db2
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># update b3:</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    b3 <span style="color:#44aadd; ">-</span><span style="color:#308080; ">=</span> lr <span style="color:#44aadd; ">*</span> db3
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># update b4:</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    b4 <span style="color:#44aadd; ">-</span><span style="color:#308080; ">=</span> lr <span style="color:#44aadd; ">*</span> db4
    
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># Store the gradients:</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    grads <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span>df1<span style="color:#308080; ">,</span> df2<span style="color:#308080; ">,</span> dw3<span style="color:#308080; ">,</span> dw4<span style="color:#308080; ">,</span> db1<span style="color:#308080; ">,</span> db2<span style="color:#308080; ">,</span> db3<span style="color:#308080; ">,</span> db4<span style="color:#308080; ">]</span> 
    
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># Store the updated params</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    params <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span>f1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> w3<span style="color:#308080; ">,</span> w4<span style="color:#308080; ">,</span> b1<span style="color:#308080; ">,</span> b2<span style="color:#308080; ">,</span> b3<span style="color:#308080; ">,</span> b4<span style="color:#308080; ">]</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> grads<span style="color:#308080; ">,</span> params<span style="color:#308080; ">,</span> loss
</pre>

### 4.4. Step 4: Train the built CNN model:

#### 4.4.1. A function for training the CNN model:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> train<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">,</span> y_train<span style="color:#308080; ">,</span> num_classes <span style="color:#308080; ">=</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> lr <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.01</span><span style="color:#308080; ">,</span> img_depth <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span>\
          f <span style="color:#308080; ">=</span> <span style="color:#008c00; ">5</span><span style="color:#308080; ">,</span> num_filt1 <span style="color:#308080; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> num_filt2 <span style="color:#308080; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> num_epochs <span style="color:#308080; ">=</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span> save_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'params.pkl'</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;A  function for training the CNN model</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    <span style="color:#595979; ">#-------------------------------------</span>
    <span style="color:#595979; "># Initializing all the parameters</span>
    <span style="color:#595979; ">#-------------------------------------</span>
    f1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> w3<span style="color:#308080; ">,</span> w4 <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>num_filt1<span style="color:#308080; ">,</span> img_depth<span style="color:#308080; ">,</span> f<span style="color:#308080; ">,</span> f<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>num_filt2<span style="color:#308080; ">,</span> num_filt1<span style="color:#308080; ">,</span> f<span style="color:#308080; ">,</span> f<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">128</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">800</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">128</span><span style="color:#308080; ">)</span>
    f1 <span style="color:#308080; ">=</span> initializeFilter<span style="color:#308080; ">(</span>f1<span style="color:#308080; ">)</span>
    f2 <span style="color:#308080; ">=</span> initializeFilter<span style="color:#308080; ">(</span>f2<span style="color:#308080; ">)</span>
    w3 <span style="color:#308080; ">=</span> initializeWeight<span style="color:#308080; ">(</span>w3<span style="color:#308080; ">)</span>
    w4 <span style="color:#308080; ">=</span> initializeWeight<span style="color:#308080; ">(</span>w4<span style="color:#308080; ">)</span>

    b1 <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>f1<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    b2 <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>f2<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    b3 <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>w3<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    b4 <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>w4<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; ">#-------------------------------------</span>
    <span style="color:#595979; "># Set the parameters</span>
    <span style="color:#595979; ">#-------------------------------------</span>
    params <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span>f1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> w3<span style="color:#308080; ">,</span> w4<span style="color:#308080; ">,</span> b1<span style="color:#308080; ">,</span> b2<span style="color:#308080; ">,</span> b3<span style="color:#308080; ">,</span> b4<span style="color:#308080; ">]</span>

    <span style="color:#595979; ">#-------------------------------------</span>
    <span style="color:#595979; "># Cost array for each epoch</span>
    <span style="color:#595979; ">#-------------------------------------</span>
    epoch_cost <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#308080; ">]</span>
    <span style="color:#595979; ">#-------------------------------------</span>
    <span style="color:#595979; "># Accuracy array for each epoch</span>
    <span style="color:#595979; ">#-------------------------------------</span>
    epoch_accuracy <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#308080; ">]</span>
    
    <span style="color:#595979; ">#-------------------------------------</span>
    <span style="color:#595979; "># Iterate over the epochs:</span>
    <span style="color:#595979; ">#-------------------------------------</span>
    <span style="color:#200080; font-weight:bold; ">for</span> epoch <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_epochs<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'---------------------------------------------------------------'</span><span style="color:#308080; ">)</span>
        <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Training Epoch # %d of %d'</span> <span style="color:#44aadd; ">%</span> <span style="color:#308080; ">(</span>epoch <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> num_epochs<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'---------------------------------------------------------------'</span><span style="color:#308080; ">)</span>

        <span style="color:#595979; ">#-------------------------------------</span>
        <span style="color:#595979; "># Shuffle the training data</span>
        <span style="color:#595979; ">#-------------------------------------</span>
        permutation <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>permutation<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        train_images <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">[</span>permutation<span style="color:#308080; ">]</span>
        train_labels <span style="color:#308080; ">=</span> y_train<span style="color:#308080; ">[</span>permutation<span style="color:#308080; ">]</span>

        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># Partial training metrics of the model: 100 images</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># compute the average loss for the 100 images</span>
        loss_100_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
        <span style="color:#595979; "># compute the average accuracy 100 images</span>
        accuracy_100_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
        
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># Training metrics of the model: For each epoch</span>
        <span style="color:#595979; ">#--------------------------------------------------------</span>
        <span style="color:#595979; "># compute the average loss for each epoch</span>
        loss_epoch <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
        <span style="color:#595979; "># compute the average accuracy for each epoch</span>
        accuracy_epoch <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>

        <span style="color:#595979; "># the numbe rof training images</span>
        num_train_images <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span>
        <span style="color:#595979; "># iterate over the training images</span>
        <span style="color:#200080; font-weight:bold; ">for</span> i<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>im<span style="color:#308080; ">,</span> label<span style="color:#308080; ">)</span> <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">enumerate</span><span style="color:#308080; ">(</span><span style="color:#400000; ">zip</span><span style="color:#308080; ">(</span>train_images<span style="color:#308080; ">,</span> train_labels<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
            <span style="color:#200080; font-weight:bold; ">if</span> i <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">99</span><span style="color:#308080; ">:</span>
                <span style="color:#595979; "># display a message</span>
                <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Processing image #: {0} of {1}'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>i<span style="color:#308080; ">,</span> num_train_images<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
                <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'---------------------------------------------------------------'</span><span style="color:#308080; ">)</span>
                <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'[Step %d]: The last 100 images: Average: Loss %.3f | Accuracy: %d%%'</span>\
                      <span style="color:#44aadd; ">%</span><span style="color:#308080; ">(</span>i <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> loss_100_images <span style="color:#44aadd; ">/</span> <span style="color:#008c00; ">100</span><span style="color:#308080; ">,</span> accuracy_100_images<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
                <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'---------------------------------------------------------------'</span><span style="color:#308080; ">)</span>
                <span style="color:#595979; "># reset the average to 0</span>
                loss_100_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
                <span style="color:#595979; "># reset the average accuracy to 0</span>
                accuracy_100_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
            
            <span style="color:#595979; ">#------------------------------------------------</span>
            <span style="color:#595979; "># Reshape the input image and its label</span>
            <span style="color:#595979; ">#------------------------------------------------</span>
            <span style="color:#595979; "># reshape to 3D shape</span>
            im <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>expand_dims<span style="color:#308080; ">(</span>im<span style="color:#308080; ">,</span> axis<span style="color:#308080; ">=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span>
            <span style="color:#595979; "># convert label to one-hot shape</span>
            label_1_hot <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>eye<span style="color:#308080; ">(</span>num_classes<span style="color:#308080; ">)</span><span style="color:#308080; ">[</span><span style="color:#400000; ">int</span><span style="color:#308080; ">(</span>label<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>num_classes<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span> 
            
            <span style="color:#595979; ">#------------------------------------------------</span>
            <span style="color:#595979; "># Call the convolution function</span>
            <span style="color:#595979; ">#------------------------------------------------</span>
            grads<span style="color:#308080; ">,</span> params<span style="color:#308080; ">,</span> loss <span style="color:#308080; ">=</span> conv_forward_backward_pass<span style="color:#308080; ">(</span>im<span style="color:#308080; ">,</span> label_1_hot<span style="color:#308080; ">,</span> params<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span> lr<span style="color:#308080; ">)</span>
            
            <span style="color:#595979; ">#------------------------------------------------</span>
            <span style="color:#595979; "># Predict the class of thetraining image using </span>
            <span style="color:#595979; "># the trained model</span>
            <span style="color:#595979; ">#------------------------------------------------</span>
            pred_class<span style="color:#308080; ">,</span> pred_prob <span style="color:#308080; ">=</span> predict<span style="color:#308080; ">(</span>im<span style="color:#308080; ">,</span> f1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> w3<span style="color:#308080; ">,</span> w4<span style="color:#308080; ">,</span> b1<span style="color:#308080; ">,</span> b2<span style="color:#308080; ">,</span> b3<span style="color:#308080; ">,</span> b4<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">)</span>
            
            <span style="color:#595979; ">#------------------------------------------------</span>
            <span style="color:#595979; "># Compute the accuracy:</span>
            <span style="color:#595979; ">#------------------------------------------------</span>
            accuracy <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span> <span style="color:#200080; font-weight:bold; ">if</span> pred_class <span style="color:#44aadd; ">==</span> label <span style="color:#200080; font-weight:bold; ">else</span> <span style="color:#008c00; ">0</span>
            
            <span style="color:#595979; ">#------------------------------------------------</span>
            <span style="color:#595979; "># Update the performance evaluation metrics for </span>
            <span style="color:#595979; "># 100 images: </span>
            <span style="color:#595979; ">#------------------------------------------------</span>
            <span style="color:#595979; "># update the average cost </span>
            loss_100_images <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> loss
            
            <span style="color:#595979; "># update the accuracy</span>
            accuracy_100_images <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> accuracy
            
            <span style="color:#595979; ">#------------------------------------------------</span>
            <span style="color:#595979; "># Update the performance evaluation metrics for </span>
            <span style="color:#595979; "># 1each epoch</span>
            <span style="color:#595979; ">#------------------------------------------------</span>
            <span style="color:#595979; "># update the average loss for each epoch</span>
            loss_epoch <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> loss
            <span style="color:#595979; "># update the average accuracy for each epoch</span>
            accuracy_epoch <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> accuracy
        

        <span style="color:#595979; ">#------------------------------------------------</span>
        <span style="color:#595979; "># update the epoch average cost </span>
        <span style="color:#595979; ">#------------------------------------------------</span>
        loss_epoch <span style="color:#44aadd; ">/</span><span style="color:#308080; ">=</span> num_train_images
        <span style="color:#595979; "># append to the cost</span>
        epoch_cost<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>loss_epoch<span style="color:#308080; ">)</span>
        
        <span style="color:#595979; ">#------------------------------------------------</span>
        <span style="color:#595979; "># update the epoch average accuracy</span>
        <span style="color:#595979; ">#------------------------------------------------</span>
        accuracy_epoch <span style="color:#44aadd; ">/</span><span style="color:#308080; ">=</span> num_train_images
        <span style="color:#595979; "># append to the accuracy</span>
        epoch_accuracy<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>accuracy_epoch<span style="color:#308080; ">)</span>
        
    <span style="color:#595979; "># append the params and loss together</span>
    to_save <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span>params<span style="color:#308080; ">,</span> epoch_cost<span style="color:#308080; ">,</span> epoch_accuracy<span style="color:#308080; ">]</span>
    <span style="color:#595979; "># save the params and loss to a file</span>
    <span style="color:#200080; font-weight:bold; ">with</span> <span style="color:#400000; ">open</span><span style="color:#308080; ">(</span>save_path<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">'wb'</span><span style="color:#308080; ">)</span> <span style="color:#200080; font-weight:bold; ">as</span> <span style="color:#400000; ">file</span><span style="color:#308080; ">:</span>
        pickle<span style="color:#308080; ">.</span>dump<span style="color:#308080; ">(</span>to_save<span style="color:#308080; ">,</span> <span style="color:#400000; ">file</span><span style="color:#308080; ">)</span>

    <span style="color:#595979; "># return the cost</span>
    <span style="color:#200080; font-weight:bold; ">return</span> epoch_cost<span style="color:#308080; ">,</span> epoch_accuracy
</pre>


#### 4.4.2. Train the CNN model:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># call the above functionality to train the model:</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># The name of the saved</span>
<span style="color:#595979; ">#------------------------------------------------</span>
save_path <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'params.pkl'</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># - The MNIST dataset has: </span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; ">#   - 60,000 training images</span>
<span style="color:#595979; ">#   - 10,000 test images</span>
<span style="color:#595979; ">#</span>
<span style="color:#595979; "># - Using the full dataset to train </span>
<span style="color:#595979; ">#   the developed CNN is time consuing</span>
<span style="color:#595979; "># - We shall only use a subset of the </span>
<span style="color:#595979; ">#   dataset.</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># the number of used training images</span>
num_used_train_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1000</span>

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Training parameters:</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># the number of classes</span>
num_classes <span style="color:#308080; ">=</span> <span style="color:#008c00; ">10</span>
<span style="color:#595979; "># the learning rate</span>
learning_rate <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.01</span>
<span style="color:#595979; "># the image size</span>
img_depth <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
<span style="color:#595979; "># the filter size</span>
f <span style="color:#308080; ">=</span> <span style="color:#008c00; ">5</span>
<span style="color:#595979; "># CONV-1: number of filters</span>
num_filt1 <span style="color:#308080; ">=</span> <span style="color:#008c00; ">8</span>
<span style="color:#595979; "># CONV-1: number of filters</span>
num_filt2 <span style="color:#308080; ">=</span> <span style="color:#008c00; ">8</span>
<span style="color:#595979; "># the number of training epochs</span>
num_epochs <span style="color:#308080; ">=</span> <span style="color:#008c00; ">5</span>

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Training data subset</span>
<span style="color:#595979; ">#------------------------------------------------</span>
x_train <span style="color:#308080; ">=</span> x_train<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span>num_used_train_images<span style="color:#308080; ">]</span>
y_train <span style="color:#308080; ">=</span> y_train<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span>num_used_train_images<span style="color:#308080; ">]</span>

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># normalize x_train</span>
<span style="color:#595979; ">#------------------------------------------------</span>
x_train <span style="color:#308080; ">=</span> x_train <span style="color:#44aadd; ">-</span> np<span style="color:#308080; ">.</span>mean<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">)</span>
x_train <span style="color:#44aadd; ">/</span><span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>std<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># reshape y_train </span>
<span style="color:#595979; ">#------------------------------------------------</span>
y_train<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_train<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># train the model</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'------------------------------------------'</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Train the CNN model:'</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'------------------------------------------'</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Number of train images = '</span><span style="color:#308080; ">,</span> num_used_train_images<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'------------------------------------------'</span><span style="color:#308080; ">)</span>
cost<span style="color:#308080; ">,</span> accuracy <span style="color:#308080; ">=</span> train<span style="color:#308080; ">(</span>x_train<span style="color:#308080; ">,</span> y_train<span style="color:#308080; ">,</span> num_classes<span style="color:#308080; ">,</span> learning_rate<span style="color:#308080; ">,</span>\
                 img_depth<span style="color:#308080; ">,</span> f<span style="color:#308080; ">,</span> num_filt1<span style="color:#308080; ">,</span> num_filt2 <span style="color:#308080; ">,</span>\
                 num_epochs<span style="color:#308080; ">,</span> save_path<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'------------------------------------------'</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'The CNN model was trained successfully!'</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'------------------------------------------'</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Plot cost the Cost function</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># set the figure size</span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">6</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># plot the cost</span>
plt<span style="color:#308080; ">.</span>plot<span style="color:#308080; ">(</span>cost<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">'r-'</span><span style="color:#308080; ">,</span> linewidth <span style="color:#308080; ">=</span> <span style="color:#008000; ">2.0</span><span style="color:#308080; ">,</span> label<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'Loss'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># xlabel</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'# Epochs'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># ylabel</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Loss'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># legend</span>
plt<span style="color:#308080; ">.</span>legend<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set grid on</span>
plt<span style="color:#308080; ">.</span>grid<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># show the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Plot cost the Accuracy function</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># set the figure size</span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">6</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># plot the accuracy</span>
plt<span style="color:#308080; ">.</span>plot<span style="color:#308080; ">(</span>accuracy<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">'g-'</span><span style="color:#308080; ">,</span> linewidth <span style="color:#308080; ">=</span> <span style="color:#008000; ">2.0</span><span style="color:#308080; ">,</span> label<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'Accuracy'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># xlabel</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'# Epochs'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># ylabel</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Accuracy'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># legend</span>
plt<span style="color:#308080; ">.</span>legend<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set the grid on</span>
plt<span style="color:#308080; ">.</span>grid<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># show the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Train the CNN model<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Number of train images <span style="color:#308080; ">=</span>  <span style="color:#008c00; ">1000</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Training Epoch <span style="color:#595979; "># 1 of 5</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Processing image <span style="color:#595979; ">#: 99 of 1000</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#308080; ">[</span>Step <span style="color:#008c00; ">100</span><span style="color:#308080; ">]</span><span style="color:#308080; ">:</span> The last <span style="color:#008c00; ">100</span> images<span style="color:#308080; ">:</span> Average<span style="color:#308080; ">:</span> Loss <span style="color:#008000; ">2.281</span> <span style="color:#44aadd; ">|</span> Accuracy<span style="color:#308080; ">:</span> <span style="color:#008c00; ">25</span><span style="color:#44aadd; ">%</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Processing image <span style="color:#595979; ">#: 199 of 1000</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#308080; ">[</span>Step <span style="color:#008c00; ">200</span><span style="color:#308080; ">]</span><span style="color:#308080; ">:</span> The last <span style="color:#008c00; ">100</span> images<span style="color:#308080; ">:</span> Average<span style="color:#308080; ">:</span> Loss <span style="color:#008000; ">2.300</span> <span style="color:#44aadd; ">|</span> Accuracy<span style="color:#308080; ">:</span> <span style="color:#008c00; ">25</span><span style="color:#44aadd; ">%</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Processing image <span style="color:#595979; ">#: 299 of 1000</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#308080; ">[</span>Step <span style="color:#008c00; ">300</span><span style="color:#308080; ">]</span><span style="color:#308080; ">:</span> The last <span style="color:#008c00; ">100</span> images<span style="color:#308080; ">:</span> Average<span style="color:#308080; ">:</span> Loss <span style="color:#008000; ">2.299</span> <span style="color:#44aadd; ">|</span> Accuracy<span style="color:#308080; ">:</span> <span style="color:#008c00; ">23</span><span style="color:#44aadd; ">%</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Processing image <span style="color:#595979; ">#: 399 of 1000</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#308080; ">[</span>Step <span style="color:#008c00; ">400</span><span style="color:#308080; ">]</span><span style="color:#308080; ">:</span> The last <span style="color:#008c00; ">100</span> images<span style="color:#308080; ">:</span> Average<span style="color:#308080; ">:</span> Loss <span style="color:#008000; ">2.264</span> <span style="color:#44aadd; ">|</span> Accuracy<span style="color:#308080; ">:</span> <span style="color:#008c00; ">49</span><span style="color:#44aadd; ">%</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Processing image <span style="color:#595979; ">#: 499 of 1000</span>
<span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>
<span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>
<span style="color:#308080; ">[</span>Step <span style="color:#008c00; ">700</span><span style="color:#308080; ">]</span><span style="color:#308080; ">:</span> The last <span style="color:#008c00; ">100</span> images<span style="color:#308080; ">:</span> Average<span style="color:#308080; ">:</span> Loss <span style="color:#008000; ">0.209</span> <span style="color:#44aadd; ">|</span> Accuracy<span style="color:#308080; ">:</span> <span style="color:#008c00; ">99</span><span style="color:#44aadd; ">%</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Processing image <span style="color:#595979; ">#: 799 of 1000</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#308080; ">[</span>Step <span style="color:#008c00; ">800</span><span style="color:#308080; ">]</span><span style="color:#308080; ">:</span> The last <span style="color:#008c00; ">100</span> images<span style="color:#308080; ">:</span> Average<span style="color:#308080; ">:</span> Loss <span style="color:#008000; ">0.090</span> <span style="color:#44aadd; ">|</span> Accuracy<span style="color:#308080; ">:</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">%</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Processing image <span style="color:#595979; ">#: 899 of 1000</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#308080; ">[</span>Step <span style="color:#008c00; ">900</span><span style="color:#308080; ">]</span><span style="color:#308080; ">:</span> The last <span style="color:#008c00; ">100</span> images<span style="color:#308080; ">:</span> Average<span style="color:#308080; ">:</span> Loss <span style="color:#008000; ">0.051</span> <span style="color:#44aadd; ">|</span> Accuracy<span style="color:#308080; ">:</span> <span style="color:#008c00; ">99</span><span style="color:#44aadd; ">%</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Processing image <span style="color:#595979; ">#: 999 of 1000</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#308080; ">[</span>Step <span style="color:#008c00; ">1000</span><span style="color:#308080; ">]</span><span style="color:#308080; ">:</span> The last <span style="color:#008c00; ">100</span> images<span style="color:#308080; ">:</span> Average<span style="color:#308080; ">:</span> Loss <span style="color:#008000; ">0.065</span> <span style="color:#44aadd; ">|</span> Accuracy<span style="color:#308080; ">:</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">%</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The CNN model was trained successfully!
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

<img src="images/Loss-Accuracy-Results.PNG" width="1000"/>

### 4.5. Step 5: Evaluate the trained CNN model:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Evaluate the trained the model:</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># - The MNIST dataset has: </span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; ">#   - 60,000 training images</span>
<span style="color:#595979; ">#   - 10,000 test images</span>
<span style="color:#595979; ">#</span>
<span style="color:#595979; "># - Using the full dataset to train </span>
<span style="color:#595979; ">#   the developed CNN is time consuing</span>
<span style="color:#595979; "># - We shall only use a subset of the </span>
<span style="color:#595979; ">#   dataset.</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># the number of used test image</span>
num_used_test_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1000</span>  

<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'------------------------------------------'</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Evaluate the performance of the trained CNN model:'</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'------------------------------------------'</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Number of test images = '</span><span style="color:#308080; ">,</span> num_used_test_images<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'------------------------------------------'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># load the trained model</span>
<span style="color:#595979; ">#------------------------------------------------</span>
params<span style="color:#308080; ">,</span> cost<span style="color:#308080; ">,</span> accuracy <span style="color:#308080; ">=</span> pickle<span style="color:#308080; ">.</span>load<span style="color:#308080; ">(</span><span style="color:#400000; ">open</span><span style="color:#308080; ">(</span>save_path<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">'rb'</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># get the model paramaters</span>
<span style="color:#308080; ">[</span>f1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> w3<span style="color:#308080; ">,</span> w4<span style="color:#308080; ">,</span> b1<span style="color:#308080; ">,</span> b2<span style="color:#308080; ">,</span> b3<span style="color:#308080; ">,</span> b4<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> params

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Get test data</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Test data subset</span>
<span style="color:#595979; ">#------------------------------------------------</span>
x_test <span style="color:#308080; ">=</span> x_test<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span>num_used_test_images<span style="color:#308080; ">]</span>
y_test <span style="color:#308080; ">=</span> y_test<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span>num_used_test_images<span style="color:#308080; ">]</span>

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># normalize x_test</span>
<span style="color:#595979; ">#------------------------------------------------</span>
x_test <span style="color:#308080; ">=</span> x_test <span style="color:#44aadd; ">-</span> np<span style="color:#308080; ">.</span>mean<span style="color:#308080; ">(</span>x_test<span style="color:#308080; ">)</span>
x_test <span style="color:#44aadd; ">/</span><span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>std<span style="color:#308080; ">(</span>x_test<span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># reshape y_train </span>
<span style="color:#595979; ">#------------------------------------------------</span>
y_test<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y_test<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>

<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># initialize the correlation</span>
<span style="color:#595979; ">#------------------------------------------------</span>
corr <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
digit_count <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#200080; font-weight:bold; ">for</span> i <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span>
digit_correct <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#200080; font-weight:bold; ">for</span> i <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span>
   
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'-----------------------------------------'</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Computing accuracy over test set:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'-----------------------------------------'</span><span style="color:#308080; ">)</span>

<span style="color:#595979; "># iterate over the test images</span>
<span style="color:#200080; font-weight:bold; ">for</span> i<span style="color:#308080; ">,</span> <span style="color:#308080; ">(</span>im<span style="color:#308080; ">,</span> label<span style="color:#308080; ">)</span> <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">enumerate</span><span style="color:#308080; ">(</span><span style="color:#400000; ">zip</span><span style="color:#308080; ">(</span>x_test<span style="color:#308080; ">,</span> y_test<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># reshape the input image and its label</span>
    <span style="color:#595979; ">#------------------------------------------------</span>
    <span style="color:#595979; "># reshape to 3D shape</span>
    im <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>expand_dims<span style="color:#308080; ">(</span>im<span style="color:#308080; ">,</span> axis<span style="color:#308080; ">=</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">)</span> 
    <span style="color:#595979; "># call the prediction function to classify the input test image</span>
    pred<span style="color:#308080; ">,</span> prob <span style="color:#308080; ">=</span> predict<span style="color:#308080; ">(</span>im<span style="color:#308080; ">,</span> f1<span style="color:#308080; ">,</span> f2<span style="color:#308080; ">,</span> w3<span style="color:#308080; ">,</span> w4<span style="color:#308080; ">,</span> b1<span style="color:#308080; ">,</span> b2<span style="color:#308080; ">,</span> b3<span style="color:#308080; ">,</span> b4<span style="color:#308080; ">)</span>
    <span style="color:#595979; "># increment the the class for this image</span>
    digit_count<span style="color:#308080; ">[</span><span style="color:#400000; ">int</span><span style="color:#308080; ">(</span>label<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span><span style="color:#008c00; ">1</span>
    <span style="color:#595979; "># check if the prediction is correct</span>
    <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span>pred <span style="color:#44aadd; ">==</span> label<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># increment the number of correct predictions</span>
        corr <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
        digit_correct<span style="color:#308080; ">[</span>pred<span style="color:#308080; ">]</span> <span style="color:#44aadd; ">+</span><span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
    <span style="color:#595979; "># display a message</span>
    <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> i <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Accuracy: %0.2f%%"</span> <span style="color:#44aadd; ">%</span> <span style="color:#308080; ">(</span><span style="color:#400000; ">float</span><span style="color:#308080; ">(</span>corr<span style="color:#44aadd; ">/</span><span style="color:#308080; ">(</span>i<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#44aadd; ">*</span><span style="color:#008c00; ">100</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'-----------------------------------------'</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Overall Accuracy: %.2f"</span> <span style="color:#44aadd; ">%</span> <span style="color:#308080; ">(</span><span style="color:#400000; ">float</span><span style="color:#308080; ">(</span>corr<span style="color:#44aadd; ">/</span>num_used_test_images<span style="color:#44aadd; ">*</span><span style="color:#008c00; ">100</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'-----------------------------------------'</span><span style="color:#308080; ">)</span>
x <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>arange<span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span>
digit_recall <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span>x<span style="color:#44aadd; ">/</span>y <span style="color:#200080; font-weight:bold; ">for</span> x<span style="color:#308080; ">,</span>y <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">zip</span><span style="color:#308080; ">(</span>digit_correct<span style="color:#308080; ">,</span> digit_count<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Digits'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Recall'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Recall on Test Set"</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>bar<span style="color:#308080; ">(</span>x<span style="color:#308080; ">,</span>digit_recall<span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Evaluate the performance of the trained CNN model<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Number of test images <span style="color:#308080; ">=</span>  <span style="color:#008c00; ">1000</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Computing accuracy over test <span style="color:#400000; ">set</span><span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Overall Accuracy<span style="color:#308080; ">:</span> <span style="color:#008000; ">97.60</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

### 4.6. Part 6: Display a final message after successful execution completion:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> 
      <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">05</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">15</span> <span style="color:#008c00; ">14</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">25</span><span style="color:#308080; ">:</span><span style="color:#008000; ">15.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>


## 5. Analysis

* In this project, we demonstrated the step-by-step implementation of a Convolutional Neural Network (CNN) from scratch to classify images of hand-written digits (0-9), using the MNIST data set. 
* We did not make use of any Deep Learning frameworks such as Tensorflow, Keras, etc. 
* The classification accuracy achieved by the implemented CNN are comparable to those obtained using Deep learning frameworks, such as Tensorflow or Keras. 
* It should be mentioned the implemented CNN is much slower, during training and inference, than using the  Tensorflow or Keras, which are optimized. 
* Implementing the CNN from scratch has helped gain valuable insights and understanding of convolutional networks. 

## 6. Future Work

* We plan to explore the following related issues:

  * To generalize the implementation of the CNN model to easily handle other CNN with different layers and structures. 


## 7. References

1. Yun Lecun, Corina Cortes, Christopher J.C. Buges. The MNIST database if handwritten digits. http://yann.lecun.com/exdb/mnist/
2. Victor Zhou. CNNs, Part 1: An Introduction to Convolutional Neural Networks A simple guide to what CNNs are, how they work, and how to build one from scratch in Python. https://victorzhou.com/blog/intro-to-cnns-part-1/ 
3. Victor Zhou. CNNs, Part 2: Training a Convolutional Neural Network A simple walkthrough of deriving backpropagation for CNNs and implementing it from scratch in Python. https://victorzhou.com/blog/intro-to-cnns-part-2/ 
4. PULKIT SHARMA. A Comprehensive Tutorial to learn Convolutional Neural Networks from Scratch. https://www.analyticsvidhya.com/blog/2018/12/guide-convolutional-neural-network-cnn/ 
 5. Satsawat Natakarnkitkul. Beginners Guide to Convolutional Neural Network from Scratch " Kuzushiji-MNIST. https://towardsai.net/p/machine-learning/beginner-guides-to-convolutional-neural-network-from-scratch-kuzushiji-mnist-75f42c175b21 
6. Alejandro Escontrela. Convolutional Neural Networks from the ground up A NumPy implementation of the famed Convolutional Neural Network: one of the most influential neural network architectures to date. https://towardsdatascience.com/convolutional-neural-networks-from-the-ground-up-c67bb41454e1
