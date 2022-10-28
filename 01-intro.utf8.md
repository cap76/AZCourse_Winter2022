---
output:
  html_document: default
  pdf_document: default
---
# Introduction {#intro}

In this workbook, we cover two themes. 

We begin in chapter \@ref(logistic-regression) with linear regression and logistic regression. Here we will explore how to use these statistical methods in the context of machine learning. The majority of the excercises are based on a plant infection dataset. We will see how regression models of different complexity can be fit to the dataset, and how held out data can be used to choose the best model. After this, we will see how logistic regression approaches can be used to pick out marker genes that indicate infected status. Finally, we explore how regression approaches can be used in the context of more recent datasets: inference of spatially-resolved gene expression patterns in a developing embryo.

In chapter \@ref(mlnn) we introduce the concepts of multilayer neural networks. We demonstrate how neural networks can be built for regression applications. Later we introduce Convolutional Neural Networks (CNN) and see how they can be used for image recognition. In this chapter we build an algorithm capable of recognising a well known cartoon characer in a set of images. Finally, we briefly discuss how these basic approaches can be built into more complex algorithms.

# Installation

For the second half of this workbook we will make use of an R wrapper for keras (a Python package, which itself backends to Tensorflow). To be able to do so, we will have to install Python (I have tested on Python 3.9), tensorflow and keras. Python can be installed from [here](https://www.python.org/downloads/). When installing Python there will be an option to add the package to the system path: choosing to do so will make things a lot easier when it comes to running the backend in R. Depending on your operating system, it may also be necessary to allow long long path names (this seems to be a Windows option). If you do not allow long path names, tensorflow may not install. If you did not enable Long paths, this will have to be set manually, and instructions can be found on Google by searching: How do I enable enabled Win32 long paths? Once Python is installed, it will be necessary to idenity how to call it from the Terminal/Command Line. Usually, if you have addedd the path on installation this will be the Python version i.e., I can launch the Python3.9 from Terminal/Command Line via:


```r
python3.9
```

This will usually work if you have multiple versions installed, but sometimes names can be mixed up. For example, I installed Python3.9 and Python 3.6 on a Windows machine, and could call the former with 


```r
python3
```

but had to call:


```r
python
```

to open Python3.6. The version of Python you have launched will usually be displayed on launch. Once you've identified how to launch the specific version of Python you want, the next step is to install tensorflow to that version. Sometimes when you have multiple installs it can be difficult to ensure the correct pip is called. To avoid this confusion, you can be explicit:


```r
python3.9 -m pip install tensorflow
```

You can be even more specific by selecting a version of tensorflow:


```r
python3.9 -m pip install tensorflow==2.7.0
```

Tensorflow 2.2.0 is the version I have installed on my machine. Keras has already been incorporated into the most recent versions of tensorflow, and so it may not be necesary to install a seperate version of keras. For debugging purposes I did not install keras. You can check things have installed within Python by launching a python instance and loading the packages:


```r
python3.9
```

Then from within Python 

```r
import tensorflow as tf
import keras

tf.version.VERSION
keras.__version__
exit()
```

Finally, we will open Rstudio, install reticulate, set the version of Python to use, and install the R backend to tensorflow (and keras depending on the verion you have installed).  A more complete installation for the R keras/tensorflow packages can be found [here](https://tensorflow.rstudio.com/install/).


```r
install.packages("reticulate")
library(reticulate)
use_python("Python3.9")
install.packages("tensorflow")
#install.packages("keras")
```

In some cases, particularly for Windows, you might have to resort to giving reticulate the full path name to the Python executable. You can find our where your Python executables are by typing in the terminal:


```r
where python
```

You should then get a full path to the various insalls and can use this in R, which will look something like:


```r
install.packages("reticulate")
library(reticulate)
use_python("C:/Program Files/Python39/python.exe")
install.packages("tensorflow")
#install.packages("keras")
```
At this stage you should now be ready to run Keras in R.

# Course materials

The course materials can be viewed online [here](https://cap76.github.io/intro-machine-learning-2021/). A complete copy of the course (with all datasets, lecture slides, and compiled materials) can be accessed via GitHub [here]([here](https://github.com/cap76/AZCourse_Winter2022). By default, Git is installed on linux and macOS. To clone a copy on macOS use terminal navigate to the directory you want to use and type:


```r
git pull https://github.com/cap76/AZCourse_Winter2022.git
```

For Windows machines you may have to [set up Git](https://www.computerhope.com/issues/ch001927.htm). Alternatively, I have uploaded a copy of the complete folder on GoogleDrive.

More information about Python installations can be found at the links below.

[Installing tensorflow/keras for R](https://tensorflow.rstudio.com/install/)

[Installing Python Linux](http://docs.python-guide.org/en/latest/starting/install3/linux/)
[Installing Python for Mac](http://docs.python-guide.org/en/latest/starting/install3/osx/)
[Installing Python via Conda](https://conda.io/docs/user-guide/tasks/manage-python.html)

[Installing Tensorflow](https://www.tensorflow.org/install/)
[Installing Keras](https://keras.io/#installation)

