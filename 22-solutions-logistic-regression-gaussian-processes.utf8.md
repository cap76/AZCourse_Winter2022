# Solutions to Chapter 4 - Linear regression and logistic regression {#solutions-logistic-regression}

Solutions to exercises of chapter \@ref(logistic-regression).

We can systematically fit a model with increasing degree and evaluate/plot the RMSE on the held out data.


```r
RMSE <- rep(NULL, 10)
lrfit1 <- train(y~poly(x,degree=1), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[1] <- lrfit1$results$RMSE
lrfit2 <- train(y~poly(x,degree=2), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[2] <- lrfit2$results$RMSE
lrfit3 <- train(y~poly(x,degree=3), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[3] <- lrfit3$results$RMSE
lrfit4 <- train(y~poly(x,degree=4), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[4] <- lrfit4$results$RMSE
lrfit5 <- train(y~poly(x,degree=5), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[5] <- lrfit5$results$RMSE
lrfit6 <- train(y~poly(x,degree=6), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[6] <- lrfit6$results$RMSE
lrfit7 <- train(y~poly(x,degree=7), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[7] <- lrfit7$results$RMSE
lrfit8 <- train(y~poly(x,degree=8), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[8] <- lrfit8$results$RMSE
lrfit9 <- train(y~poly(x,degree=9), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[9] <- lrfit9$results$RMSE
lrfit10 <- train(y~poly(x,degree=10), data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
RMSE[10] <- lrfit10$results$RMSE

plot(RMSE)
plot(RMSE[1:5])
```
From these plots it looks like the best model is one with degree $d=2$ or $d=4$, suggesting there is a lot more complexity to this gene. You can clean the code up to make it run in a loop. Hint: you can not directly pass a variable over to poly (y~poly(x,i) will not work) and will have to convert to a function:


```r
setdegree <- 5
f <- bquote( y~poly(x,degree=.(setdegree) ) )
lrfit11 <- train( as.formula(f) , data=data.frame(x=D[1:24,1],y=D[1:24,geneindex]), method = "lm")
```

Excercise 1.1: Here we simply specify two three models: the first a regression run on the union of the data represents the case of no DE; the second is two independent models, one for each time series. In cases where there is no DE, two independenet models will not be necessary to describe the data compared to independent ones, and the mean square error will be similar. See also [[@stegle2010robust]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3198888/).

Excercise 1.2: We have time series, we are interested in inferring the regultors of a particular gene. We can therefore regress the time seris of the gene of interest  (at time point $2$ through $T$) against combinations of putative regulators at the previous time point  (at time point $1$ through $T-1$), and use the an appropriate metric to select the optimal combinations. We can do this in parallel for all genes to arrive at a network. For further details see e.g., [[@penfold2011infer]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3262295/) and [[@penfold2019inferring]](https://pubmed.ncbi.nlm.nih.gov/30547404/)


# Solutions to Chapter 5 - Neural Networks {#solutions-nnet}

Excersie 2.1: We can simply update the input dimension and output dimensions:


```r
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(5)) %>% 
  layer_dense(units = 10, activation = "relu") %>% 
  layer_dense(units = 20, activation = "relu") %>% 
  layer_dense(3, activation = "linear")
```

Excercsie 2.2: The network architecture should be fine for this task. However a noisy version of the input data will have to be generated (e.g., by setting a random set of pixels to zero) to be passed in to the AE. A clean version of the data should be retained and passed to the AE as the output. 


```r
model = load_model_hdf5('data/RickandMorty/data/models/modelCNNTF.h5')

tf$compat$v1$disable_eager_execution()

layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)

choice = 1
activations <- activation_model %>% predict( array_reshape( allTFX_test[choice, ,] , c(1,200, 4) ) )
first_layer_activation <- activations[[1]]
```


```r
op <- par(mfrow=c(1,1))
image((first_layer_activation[1,,]), axes = FALSE )
par(op)
```

For a given sequence we can get an idea of the importance of any given basepair by artifically setting that basepair to $[0,0,0,0]$ and see how that effects the probabilty of mapping to the correct class. Let's pick a particular sequence and take a look:


```r
model = load_model_hdf5('data/RickandMorty/data/models/modelCNNTF.h5')

seqchoice <- 1

pr1 <- model %>% predict(array_reshape(allTFX_test[seqchoice, ,],c(1,200,4) ))

Delta  <- array(0, dim=c(200,200,4))
for (i in 1:200){
  Delta[i,,] <- allTFX_test[seqchoice, ,]
  Delta[i,i,1:4] <- 0
}
pr <- model %>% predict( Delta )

DeltaP <- pr1[1,1] - pr[,1]

ggplot(data.frame(x=seq(1,200,1),y=DeltaP ), aes(x = x, y = y)) + geom_line(size = 1) + geom_point(color='blue')  + theme_bw()
```

so we can see a peak region somewhere betwee 125 and 150

```r
ggplot(data.frame(x=seq(128,138,1),y=DeltaP[128:138] ), aes(x = x, y = y)) + geom_line(size = 1) + geom_point(color='blue')  + theme_bw()
```

Let's take a look at the sequence here:


```r
#BiocManager::install("motifStack")
#BiocManager::install("universalmotif")
library(motifStack)
library(universalmotif)
motif <- allTFX_test[seqchoice,128:138 ,]
colnames(motif) <- c("A","C","G","T")
motif

reversemotif <- motif
reversemotif[,c("A")] <- motif[,c("T")] 
reversemotif[,c("T")] <- motif[,c("A")]
reversemotif[,c("C")] <- motif[,c("G")]
reversemotif[,c("G")] <- motif[,c("C")]

motif<-new("pfm", mat=as.matrix(t(motif) ), name="CAP", color=colorset(alphabet="DNA",colorScheme="basepairing"))
reversemotif<-new("pfm", mat=as.matrix(t(reversemotif) ), name="CAP", color=colorset(alphabet="DNA",colorScheme="basepairing"))

Sox17pwm <- t(matrix( 
   c(7,8,3,30,0,0,0,0,0,
     9,8,18,0,1,0,0,0,	
     17,6,4,1,0,0,0,31,2,10,
     9,11,9,1,30,31,0,29,4), nrow=4, ncol=9,  byrow = TRUE))
colnames(Sox17pwm) <- c("A","C","G","T")
Sox17pwm<-new("pfm", mat=as.matrix(t(Sox17pwm) ), name="CAP", color=colorset(alphabet="DNA",colorScheme="basepairing"))


op <- par(mfrow=c(1,3))
view_motifs(Sox17pwm, use.type = "PPM")
view_motifs(motif, use.type = "PPM")
view_motifs(reversemotif, use.type = "PPM")
par(op)
```

Although these approaches are no longer considered state of the art, they still have some practical value, and have been incorporated into more complex arcitectures which, e.g., combined CNN to learn motifs with LSTM to learn long range interactions. 



```r
model <- keras_model_sequential() 

  forward_layer = layer_lstm(units = 1024, return_sequences=TRUE)
  backward_layer = layer_lstm(units = 1024, activation='relu', return_sequences=TRUE,go_backwards=TRUE)

model %>%
  layer_conv_1d(input_shape = list(200,4), filters = 1024, kernel_size = c(30)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_1d(pool_size=c(15),strides = c(4)) %>%
  layer_dropout(rate = 0.2) %>%
  bidirectional(layer = forward_layer,backward_layer=backward_layer) %>%
  layer_flatten( ) %>%
  layer_dropout(rate = 0.5) %>%
  layer_flatten( ) %>%
  layer_dense(units=100) %>%
  layer_dense(units = 3, activation = "sigmoid")

cp_callback <- callback_model_checkpoint(filepath = 'data/RickandMorty/data/models/modelCNNRNN.h5',save_weights_only = FALSE, mode = "auto",  monitor = "val_categorical_accuracy", verbose = 0)

model %>% compile(loss = "categorical_crossentropy", optimizer = "adadelta", metrics = "categorical_accuracy")

tensorflow::set_random_seed(42)
model %>% fit(x = allTFX_train, y = allYtrain , validation_data = list(allTFX_test, allYtest), epochs = 300, batch_size=1000, verbose = 2, callbacks = list(cp_callback))
```

Excercise 2.3: read images direct from their folder. 


```r
#Number of files in total around 5257, we will split roughly by
number_of_train_samples <- 4000
number_of_val_samples <- 1257
batch_size = 100

steps_per_epoch = ceiling(number_of_train_samples / batch_size)
val_steps = ceiling(number_of_val_samples / batch_size)

datagen <- image_data_generator(horizontal_flip = TRUE, validation_split = 0.2)

test_generator = flow_images_from_directory("data/RickandMorty/altdata/", target_size=c(90, 160), batch_size = batch_size, class_mode='binary',shuffle=FALSE, seed=10, subset = 'validation', color_mode = 'rgb',generator = datagen)

train_generator = flow_images_from_directory("data/RickandMorty/altdata/", target_size=c(90, 160), batch_size = batch_size, class_mode='binary',shuffle=FALSE, seed=10, subset = 'training', color_mode = 'rgb',generator = datagen)

model <- keras_model_sequential() %>%
  layer_conv_2d(input_shape = list(90,160,3), filters = 20, kernel_size = c(5,5)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters = 20, kernel_size = c(5,5)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(5,5)) %>%
  layer_activation("relu") %>%
  layer_max_pooling_2d(pool_size=c(2,2)) %>%
  layer_flatten( ) %>%
  layer_dense(units=100) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units=1, activation = "sigmoid")

cp_callback <- callback_model_checkpoint(filepath = 'data/RickandMorty/data/models/modelCNNFlowfromfolder.h5',save_weights_only = FALSE, mode = "auto",  monitor = "val_binary_accuracy", verbose = 0)

model %>% compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "binary_accuracy")

tensorflow::set_random_seed(42)
  model %>% fit(train_generator, steps_per_epoch= steps_per_epoch, validation_data = test_generator, validation_steps= val_steps, epochs = 5, verbose = 2, callbacks = list(cp_callback))
```


Excercise 2.4: The same sippet of code should be usable from the image analyses, with minor changes to "image size". We first randomly set a certain fraction of pixels to 0.


```r
cleanX <- valX
noiseX <- valX
fraction_of_pixels <- 0.25

for (i in 1:dim(noiseX)[1]) {
  Npix = prod(dim(noiseX)[2:3])
  Rpix = sample(Npix, fraction_of_pixels * Npix) 
  
  R <- noiseX[i,,,1]
  G <- noiseX[i,,,2]
  B <- noiseX[i,,,3]
  R[Rpix] <- 0
  G[Rpix] <- 0
  B[Rpix] <- 0
  
  noiseX[i,,,1] = R
  noiseX[i,,,2] = G
  noiseX[i,,,3] = B
}


grid::grid.newpage()
grid.raster(noiseX[1,1:90,1:160,1:3], interpolate=FALSE, width = 0.3, x = 0.5, y=0.2)
grid.raster(cleanX[1,1:90,1:160,1:3], interpolate=FALSE, width = 0.3, x = 0.5, y=0.5)
```
We can the train on the autoencoder to denoise the image. 


```r
model <- keras_model_sequential() %>%
  layer_conv_2d(input_shape = list(90,160,3), filters = 20, kernel_size = c(5,5)) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filters = 20, kernel_size = c(5,5)) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filters = 64, kernel_size = c(5,5)) %>%
  layer_activation("relu") %>%
  layer_conv_2d_transpose(filters = 64, kernel_size = c(5,5)) %>%
  layer_activation("relu") %>%
  layer_conv_2d_transpose(filters = 20, kernel_size = c(5,5)) %>%
  layer_activation("relu") %>%
  layer_conv_2d_transpose(filters = 20, kernel_size = c(5,5)) %>%
  layer_activation("relu") %>%
  layer_conv_2d(filters = 3, kernel_size = c(5,5), padding = 'same') %>%
  layer_activation("sigmoid")

cp_callback <- callback_model_checkpoint(filepath = 'data/RickandMorty/data/models/modelAEND_rerun.h5',save_weights_only = FALSE, mode = "auto",  monitor = "val_mse", verbose = 0)

model %>% compile(loss = "binary_crossentropy", optimizer = "adam", metrics = "mse")

tensorflow::set_random_seed(42)
model %>% fit(x = noiseX, y = cleanX, validation_split=0.25, epochs = 5, verbose = 2, callbacks = list(cp_callback))
```

Fortunately I've already run this model for 50 epochs. We can therefore load it in and use to visualise the de-noising. The bottom image is the clean data (our gold standard), the second is the noisy image, and the third one is the de-noised version of that image. 


```r
model = load_model_hdf5('data/RickandMorty/data/models/modelAEND.h5')


cleanvalX <- trainX
noisevalX <- trainX
fraction_of_pixels <- 0.25

#Generate 100 noisy cases for testing
for (i in 1:100) {
  Npix = prod(dim(noisevalX)[2:3])
  Rpix = sample(Npix, fraction_of_pixels * Npix) 
  
  R <- noisevalX[i,,,1]
  G <- noisevalX[i,,,2]
  B <- noisevalX[i,,,3]
  R[Rpix] <- 0
  G[Rpix] <- 0
  B[Rpix] <- 0
  
  noisevalX[i,,,1] = R
  noisevalX[i,,,2] = G
  noisevalX[i,,,3] = B
}

predictAEX <- model %>% predict(noisevalX[1:100,,,])

grid::grid.newpage()
grid.raster(trainX[1,1:90,1:160,1:3], interpolate=FALSE, width = 0.3, x = 0.5, y=0.2)
grid.raster(noisevalX[1,1:90,1:160,1:3], interpolate=FALSE, width = 0.3, x = 0.5, y=0.5)
grid.raster(predictAEX[1,1:90,1:160,1:3], interpolate=FALSE, width = 0.3, x = 0.5, y=0.8)
```


# Gaussian process regression {#gaussian-process-regression}

In the previous section we briefly explored fitting multiple polynomials to our data. However, we still had to decide on the order of the polynomial beforehand. A far more powerful approach is Gaussian processes (GP) regression [[@Williams2006]](https://gaussianprocess.org/gpml/). Gaussian process regression represent a Bayesian nonparametric approach to regression capable of inferring nonlinear functions from a set of observations. Within a GP regression setting we assume the following model for the data:

$y = f(\mathbf{X})$

where $f(\cdot)$ represents an unknown nonlinear function. 

Formally, Gaussian processes are defined as a *collections of random variables, any finite subset of which are jointly Gaussian distributed* [@Williams2006]. The significance of this might not be immediately clear, and another way to think of GPs is as an infinite dimensional extension to the standard multivariate normal distribution. In the same way a Gaussian distribution is defined by its mean, $\mathbf{\mu}$, and covaraiance matrix, $\mathbf{K}$, a Gaussian processes is completely defined by its *mean function*, $m(X)$, and *covariance function*, $k(X,X^\prime)$, and we use the notation $f(x) \sim \mathcal{GP}(m(x), k(x,x^\prime))$ to denote that $f(X)$ is drawn from a Gaussian process prior.

As it is an infinite dimensional object, dealing directly with the GP prior is not feasible. However, we can make good use of the properties of a Gaussian distributions to sidestep this. Notably, the integral of a Gaussian distribution is itself a Gaussian distribution, which means that if we had a two-dimensional Gaussian distribution (defined over an x-axis and y-axis), we could integrate out the effect of y-axis to give us a (Gaussian) distribution over the x-axis. Gaussian processes share this property, which means that if we are interested only in the distribution of the function at a set of locations, $\mathbf{X}$ and $\mathbf{X}^*$, we can specify the distribution of the function over the entirity of the input domain (all of x), and analytically integrate out the effect at all other locations. This induces a natural prior distribution over the output variable that is, itself, Gaussian:

$$
\begin{eqnarray*}
\begin{pmatrix}\mathbf{y}^\top\\
\mathbf{y^*}^\top
\end{pmatrix} & \sim & N\left(\left[\begin{array}{c}
\mathbf{0}\\
\mathbf{0}\\
\end{array}\right],\left[\begin{array}{ccc}
K(\mathbf{x},\mathbf{x}) & K(\mathbf{x},\mathbf{x}^*)\\
K(\mathbf{x}^*,\mathbf{x}) & K(\mathbf{x}^*,\mathbf{x}^*) \\
\end{array}\right)\right]
\end{eqnarray*} 
$$

Quite often we deal with noisy data where:

$y = f(\mathbf{x}) + \varepsilon$,

and $\varepsilon$ represents independent Gaussian noise. In this setting we are interested in inferring the function $\mathbf{f}^*$ at $\mathbf{X}*$ i.e., using the noise corrupted data to infer the underlying function, $f(\cdot)$. To do so we note that *a priori* we have the following joint distribution:

$$
\begin{eqnarray*}
\begin{pmatrix}\mathbf{y}^\top\\
\mathbf{f^*}^\top
\end{pmatrix} & \sim & N\left(\left[\begin{array}{c}
\mathbf{0}\\
\mathbf{0}\\
\end{array}\right],\left[\begin{array}{ccc}
K(\mathbf{x},\mathbf{x})+\sigma_n^2 \mathbb{I} & K(\mathbf{x},\mathbf{x}^*)\\
K(\mathbf{x}^*,\mathbf{x}) & K(\mathbf{x}^*,\mathbf{x}^*) \\
\end{array}\right)\right]
\end{eqnarray*} 
$$

#### Sampling from the prior

In the examples below we start by sampling from a GP prior as a way of illustrating what it is that we're actualy doing. We first require a number of packages:


```r
require(MASS)
require(plyr)
require(reshape2)
require(ggplot2)
```

Recall that the GP is completely defined by its *mean function* and *covariance function*. We can assume a zero-mean function without loss of generality. Until this point, we have not said much about what the covariance function is. In general, the covariance function encodes all information about the *type* of functions we're interested in: is it smooth? Periodic? Does it have more complex structure? Does it branching? A good starting point, and the most commonly used covariance function, is the squared exponential covariance function:

$k(X,X^\prime) = \sigma^2 \exp\biggl{(}\frac{(X-X^\prime)^2}{2l^2}\biggr{)}$.

This encodes for smooth functions (functions that are infinitely differentiable), and has two hyperparameters: a length-scale hyperparameter $l$, which defines how fast the functions change over the input space (in our example this would *time*), and a process variance hyperparameter, $\sigma$, which encodes the amplitude of the function (in our examples this represents roughly the amplitude of gene expression levels). In the snippet of code, below, we implement a squared exponential covariance function


```r
covSE <- function(X1,X2,l=1,sig=1) {
  K <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  for (i in 1:nrow(K)) {
    for (j in 1:ncol(K)) {
      K[i,j] <- sig^2*exp(-0.5*(abs(X1[i]-X2[j]))^2 /l^2)
    }
  }
  return(K)
}
```

To get an idea of what this means, we can generate samples from the GP prior at a set of defined positions along $X$. Recall that due to the nature of GPs this is Gaussian distributed:


```r
x.star <- seq(-5,5,len=500) ####Define a set of points at which to evaluate the functions
sigma  <- covSE(x.star,x.star) ###Evaluate the covariance function at those locations, to give the covariance matrix.
y1 <- mvrnorm(1, rep(0, length(x.star)), sigma)
y2 <- mvrnorm(1, rep(0, length(x.star)), sigma)
y3 <- mvrnorm(1, rep(0, length(x.star)), sigma)
plot(y1,type = 'l',ylim=c(min(y1,y2,y3),max(y1,y2,y3)))
lines(y2)
lines(y3)
```

When we specify a GP, we are essentially encoding a distribution over a whole set of functions. Exactly how those functions behave depends upon the choice of covariance function and the hyperparameters. To get a feel for this, try changing the hyperparameters in the above code. What do the functions look like? A variety of other covariance functions exist, and can be found, with examples in the [Kernel Cookbook](http://www.cs.toronto.edu/~duvenaud/cookbook/).

Exercise 9.4 (optional): Try implementing another covariance function from the [Kernel Cookbook](http://www.cs.toronto.edu/~duvenaud/cookbook/) and generating samples from the GP prior. Since we have already seen that some of our genes are circadian, a useuful covariance function to try would be the periodic covariance function.

#### Inference with GPs

We can generate samples from the GP prior, but what about inference? In linear regression we aimed to infer the parameters, $m$ and $a$. What is the GP doing during inference? Essentially, it's representing the (unknown) function in terms of the observed data and the hyperparameters. Another way to look at it is that we have specified a prior distribution (encoding for all functions of a particular kind) over the input space; during inference in the noise-free case, we then discard all functions that don't pass through those observations. During inference for noisy data we assign greater weight to those functions that pass close to our observed datapoints. Essentially we're using the data to pin down a subset of the prior functions that behave in the appropriate way.

For the purpose of inference, we typically have a set of observations, $\mathbf{X}$, and outputs $\mathbf{y}$, and are interested in inferring the (unnoisy) values, $\mathbf{f}^*$, at new set of test locations, $\mathbf{X}^*$. We can infer a posterior distribution for $\mathbf{f}^*$ using Bayes' rule:

$p(\mathbf{f}^* | \mathbf{X}, \mathbf{y}, \mathbf{X}^*) = \frac{p(\mathbf{y}, \mathbf{f}^* | \mathbf{X}, \mathbf{X}^*)}{p(\mathbf{y}|\mathbf{X})}.$

A key advantage of GPs is that the preditive distribution is analytically tractible and has the following Gaussian form:

$\mathbf{f}^* | \mathbf{X}, \mathbf{y}, \mathbf{X}* \sim \mathcal{N}(\hat{f}^*,\hat{K}^*)$

where,

$\hat{f}^* = K(\mathbf{X},\mathbf{X}^*)^\top(K(\mathbf{X},\mathbf{X})+\sigma^2\mathbb{I})^{-1} \mathbf{y}$,

$\hat{K}^* = K(\mathbf{X}^*,\mathbf{X}^*)^{-1} - K(\mathbf{X},\mathbf{X}^*)^\top (K(\mathbf{X},\mathbf{X})+\sigma^2\mathbb{I})^{-1} K(\mathbf{X},\mathbf{X}^*)$.

To demonstrate this, let's assume we have an unknown function we want to infer. In our example, for data generation, we will assume this to be $y = \sin(X)$ as an illustrative example of a nonlinear function (although we know this, the GP will only ever see samples from this function, never the function itself). We might have some observations from this function at a set of input positions $X$ e.g., one observation at $x=-2$:


```r
f <- data.frame(x=c(-2),
                y=sin(c(-2)))
```

We can infer a posterior GP (and plot this against the true underlying function in red):


```r
x <- f$x
k.xx <- covSE(x,x)
k.xxs <- covSE(x,x.star)
k.xsx <- covSE(x.star,x)
k.xsxs <- covSE(x.star,x.star)

f.star.bar <- k.xsx%*%solve(k.xx)%*%f$y  ###Mean
cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx)%*%k.xxs ###Var

plot(x.star,sin(x.star),type = 'l',col="red",ylim=c(-2.2, 2.2))
points(f,type='o')
lines(x.star,f.star.bar,type = 'l')
lines(x.star,f.star.bar+2*sqrt(diag(cov.f.star)),type = 'l',pch=22, lty=2, col="black")
lines(x.star,f.star.bar-2*sqrt(diag(cov.f.star)),type = 'l',pch=22, lty=2, col="black")
```

We can see that the GP has pinned down functions that pass close to the datapoint. Of course, at this stage, the fit is not particularly good, but that's not surprising as we only had one observation. Crucially, we can see that the GP encodes the idea of *uncertainty*. Although the model fit is not particularly good, we can see exactly *where* it is no good.

Exercise 9.5 (optional): Try plotting some sample function from the posterior GP. Hint: these will be Gaussian distributed with mean {f.star.bar} and covariance {cov.f.star}.

Let's start by adding more observations. Here's what the posterior fit looks like if we include 4 observations (at $x \in [-4,-2,0,1]$):


```r
f <- data.frame(x=c(-4,-2,0,1),
                y=sin(c(-4,-2,0,1)))
x <- f$x
k.xx <- covSE(x,x)
k.xxs <- covSE(x,x.star)
k.xsx <- covSE(x.star,x)
k.xsxs <- covSE(x.star,x.star)

f.star.bar <- k.xsx%*%solve(k.xx)%*%f$y  ###Mean
cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx)%*%k.xxs ###Var

plot(x.star,sin(x.star),type = 'l',col="red",ylim=c(-2.2, 2.2))
points(f,type='o')
lines(x.star,f.star.bar,type = 'l')
lines(x.star,f.star.bar+2*sqrt(diag(cov.f.star)),type = 'l',pch=22, lty=2, col="black")
lines(x.star,f.star.bar-2*sqrt(diag(cov.f.star)),type = 'l',pch=22, lty=2, col="black")
```

And with $7$ observations:


```r
f <- data.frame(x=c(-4,-3,-2,-1,0,1,2),
                y=sin(c(-4,-3,-2,-1,0,1,2)))
x <- f$x
k.xx <- covSE(x,x)
k.xxs <- covSE(x,x.star)
k.xsx <- covSE(x.star,x)
k.xsxs <- covSE(x.star,x.star)

f.star.bar <- k.xsx%*%solve(k.xx)%*%f$y  ###Mean
cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx)%*%k.xxs ###Var

plot(x.star,sin(x.star),type = 'l',col="red",ylim=c(-2.2, 2.2))
points(f,type='o')
lines(x.star,f.star.bar,type = 'l')
lines(x.star,f.star.bar+2*sqrt(diag(cov.f.star)),type = 'l',pch=22, lty=2, col="black")
lines(x.star,f.star.bar-2*sqrt(diag(cov.f.star)),type = 'l',pch=22, lty=2, col="black")
```

We can see that with $7$ observations the posterior GP has begun to resemble the true (nonlinear) function very well: the mean of the GP lies very close to the true function and, perhaps more importantly, we continue to have an treatment for the uncertainty. 

#### Marginal Likelihood and Optimisation of Hyperparameters

Another key aspect of GP regression is the ability to analytically evaluate the marginal likelihood, otherwise referred to as the "model evidence". The marginal likelihood is the probability of generating the observed datasets under the specified prior. For a GP this would be the probability of seeing the observations $\mathbf{X}$ under a Gaussian distribtion, $\mathcal{N}(\mathbf{0},K(\mathbf{X},\mathbf{X}))$. The log marginal likelihood for a noise-free model is:

$\ln p(\mathbf{y}|\mathbf{X}) = -\frac{1}{2}\mathbf{y}^\top [K(\mathbf{X},\mathbf{X})+\sigma_n^2\mathbb{I}]^{-1} \mathbf{y} -\frac{1}{2} \ln |K(\mathbf{X},\mathbf{X})+\sigma_n^2\mathbb{I}| - \frac{n}{2}\ln 2\pi$

We calculate this in the snippet of code, below, hard-coding a small amount of Gaussian noise:


```r
calcML <- function(f,l=1,sig=1) {
  f2 <- t(f)
  yt <- f2[2,]
  y  <- f[,2]
  K <- covSE(f[,1],f[,1],l,sig)
  ML <- -0.5*yt%*%ginv(K+0.1^2*diag(length(y)))%*%y -0.5*log(det(K)) -(length(f[,1])/2)*log(2*pi);
  return(ML)
}
```

The ability to calculate the marginal likelihood gives us a way to automatically select the *hyperparameters*. We can increment hyperparameters over a range of values, and choose the values that yield the greatest marginal likelihood. In the example, below, we increment both the length-scale and process variance hyperparameter:


```r
library(plot3D)

par <- seq(.1,10,by=0.1)
ML <- matrix(rep(0, length(par)^2), nrow=length(par), ncol=length(par))
for(i in 1:length(par)) {
  for(j in 1:length(par)) {
    ML[i,j] <- calcML(f,par[i],par[j])
  }
}
persp3D(z = ML,theta = 120)
ind<-which(ML==max(ML), arr.ind=TRUE)
print(c("length-scale", par[ind[1]]))
print(c("process variance", par[ind[2]]))
```

Here we have performed a grid search to identify the optimal hyperparameters. In practice, the derivative of the marginal likelihood with respect to the hyperparameters is analytically tractable, allowing us to optimise using gradient search algorithms.


Exercise 9.7: Now try fitting a Gaussian process to one of the gene expression profiles in the Botrytis dataset. Hint: You may need to normalise the time axis. Since this data also contains a high level of noise you will also need to use a covariance function/ML calculation that incorporates noise. The snippet of code, below, does this, with the noise now representing a $3$rd hyperparameter.


```r
covSEn <- function(X1,X2,l=1,sig=1,sigman=0.1) {
  K <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  for (i in 1:nrow(K)) {
    for (j in 1:ncol(K)) {
      
      K[i,j] <- sig^2*exp(-0.5*(abs(X1[i]-X2[j]))^2 /l^2)
      
      if (i==j){
      K[i,j] <- K[i,j] + sigman^2
      }
      
    }
  }
  return(K)
}
```


```r
calcMLn <- function(f,l=1,sig=1,sigman=0.1) {
  f2 <- t(f)
  yt <- f2[2,]
  y  <- f[,2]
  K <- covSE(f[,1],f[,1],l,sig)
  ML <- -0.5*yt%*%ginv(K+diag(length(y))*sigman^2)%*%y -0.5*log(det(K+diag(length(y))*sigman^2)) -(length(f[,1])/2)*log(2*pi);
  return(ML)
}
```

#### Model Selection {#model-selection}

As well as being a useful criterion for selecting hyperparameters, the marginal likelihood can be used as a basis for selecting models. For example, we might be interested in comparing how well we fit the data using two different covariance functions: a squared exponential covariance function (model 1, $M_1$) versus a periodic covariance function (model 2, $M_2$). By taking the ratio of the marginal likelihoods we can calculate the [Bayes' Factor](https://en.wikipedia.org/wiki/Bayes_factor) (BF) which allows us to determine which model is the best:

$\mbox{BF} = \frac{ML(M_1)}{ML(M_2)}$.

High values for the BF indicate strong evidence for $M_1$ over $M_2$, whilst low values would indicate the contrary.

Excercise 3.1: Using our previous example, $y = sin(x)$ try fitting a periodic covariance function. How well does it generalise e.g., how well does it fit $f(\cdot)$ far from the observation data? How does this compare to a squared-exponential?

Example covariance functions implemented from the [Kernel Cookbook](http://www.cs.toronto.edu/~duvenaud/cookbook/). Here we implement a rational quadratic covariance function:


```r
covRQ <- function(X1,X2,l=1,sig=1,a=2) {
  K <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  for (i in 1:nrow(K)) {
    for (j in 1:ncol(K)) {
      K[i,j] <- sig^2*(1 + (abs(X1[i]-X2[j])^2/(2*a*l^2))    )^a 
    }
  }
  return(K)
}
```

Here we implement a periodic covariance function:


```r
covPer <- function(X1,X2,l=1,sig=1,p=1) {
  K <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  for (i in 1:nrow(K)) {
    for (j in 1:ncol(K)) {
      K[i,j] <- sig^2*exp(sin(pi*abs(X1[i]-X2[j])/p)^2 / l^2) 
    }
  }
  return(K)
}
```

We need to borrow the following snippets of code from the main text.


```r
require(MASS)
require(plyr)
require(reshape2)
require(ggplot2)

covSE <- function(X1,X2,l=1,sig=1) {
  K <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  for (i in 1:nrow(K)) {
    for (j in 1:ncol(K)) {
      K[i,j] <- sig^2*exp(-0.5*(abs(X1[i]-X2[j]))^2 /l^2)
    }
  }
  return(K)
}
```


```r
x.star <- seq(-5,5,len=500)
f <- data.frame(x=c(-4,-3,-2,-1,0,1,2),
                y=sin(c(-4,-3,-2,-1,0,1,2)))
x <- f$x
k.xx <- covSE(x,x)
k.xxs <- covSE(x,x.star)
k.xsx <- covSE(x.star,x)
k.xsxs <- covSE(x.star,x.star)

f.star.bar <- k.xsx%*%solve(k.xx)%*%f$y  #Mean
cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx)%*%k.xxs #Var

y1 <- mvrnorm(1, f.star.bar, cov.f.star)
y2 <- mvrnorm(1, f.star.bar, cov.f.star)
y3 <- mvrnorm(1, f.star.bar, cov.f.star)
plot(x.star,sin(x.star),type = 'l',col="red",ylim=c(-2.2, 2.2))
points(f,type = 'p',col="blue")
lines(x.star,y1,type = 'l',col="blue")
lines(x.star,y2,type = 'l',col="blue")
lines(x.star,y3,type = 'l',col="blue")
```


```r
calcML <- function(f,l=1,sig=1) {
  f2 <- t(f)
  yt <- f2[2,]
  y  <- f[,2]
  K <- covSE(f[,1],f[,1],l,sig)
  ML <- -0.5*yt%*%ginv(K+0.1^2*diag(length(y)))%*%y -0.5*log(det(K)) -(length(f[,1])/2)*log(2*pi);
  return(ML)
}
```


```r
#install.packages("plot3D")
library(plot3D)

par <- seq(.1,10,by=0.1)
ML <- matrix(rep(0, length(par)^2), nrow=length(par), ncol=length(par))
for(i in 1:length(par)) {
  for(j in 1:length(par)) {
    ML[i,j] <- calcML(f,par[i],par[j])
  }
}

ind<-which(ML==max(ML), arr.ind=TRUE)
lmap<-par[ind[1]]
varmap<-par[ind[2]]
```



```r
x.star <- seq(-5,5,len=500)
f <- data.frame(x=c(-4,-3,-2,-1,0,1,2),
                y=sin(c(-4,-3,-2,-1,0,1,2)))
x <- f$x
k.xx <- covSE(x,x,lmap,varmap)
k.xxs <- covSE(x,x.star,lmap,varmap)
k.xsx <- covSE(x.star,x,lmap,varmap)
k.xsxs <- covSE(x.star,x.star,lmap,varmap)

f.star.bar <- k.xsx%*%solve(k.xx)%*%f$y  #Mean
cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx)%*%k.xxs #Var

plot(x.star,sin(x.star),type = 'l',col="red",ylim=c(-2.2, 2.2))
points(f,type='o')
lines(x.star,f.star.bar,type = 'l')
lines(x.star,f.star.bar+2*sqrt(diag(cov.f.star)),type = 'l',pch=22, lty=2, col="black")
lines(x.star,f.star.bar-2*sqrt(diag(cov.f.star)),type = 'l',pch=22, lty=2, col="black")
```

Now try fitting a Gaussian process to one of the gene expression profiles in the Botrytis dataset.


```r
covSEn <- function(X1,X2,l=1,sig=1,sigman=0.1) {
  K <- matrix(rep(0, length(X1)*length(X2)), nrow=length(X1))
  for (i in 1:nrow(K)) {
    for (j in 1:ncol(K)) {
      
      K[i,j] <- sig^2*exp(-0.5*(abs(X1[i]-X2[j]))^2 /l^2)
      
      if (i==j){
      K[i,j] <- K[i,j] + sigman^2
      }
      
    }
  }
  return(K)
}
```


```r
geneindex <- 36
lmap <- 0.1
varmap <- 5
x.star <- seq(0,1,len=500)
f <- data.frame(x=D[25:nrow(D),1]/48, y=D[25:nrow(D),geneindex])
x <- f$x
k.xx <- covSEn(x,x,lmap,varmap,0.2)
k.xxs <- covSEn(x,x.star,lmap,varmap,0.2)
k.xsx <- covSEn(x.star,x,lmap,varmap,0.2)
k.xsxs <- covSEn(x.star,x.star,lmap,varmap,0.2)

f.star.bar <- k.xsx%*%solve(k.xx)%*%f$y  #Mean
cov.f.star <- k.xsxs - k.xsx%*%solve(k.xx)%*%k.xxs #Var

plot(f,type = 'l',col="red")
points(f,type='o')
lines(x.star,f.star.bar,type = 'l')
lines(x.star,f.star.bar+2*sqrt(diag(cov.f.star)),type = 'l',pch=22, lty=2, col="black")
lines(x.star,f.star.bar-2*sqrt(diag(cov.f.star)),type = 'l',pch=22, lty=2, col="black")
```



```r
calcMLn <- function(f,l=1,sig=1,sigman=0.1) {
  f2 <- t(f)
  yt <- f2[2,]
  y  <- f[,2]
  K <- covSE(f[,1],f[,1],l,sig)
  ML <- -0.5*yt%*%ginv(K+diag(length(y))*sigman^2)%*%y -0.5*log(det(K+diag(length(y))*sigman^2)) -(length(f[,1])/2)*log(2*pi);
  return(ML)
}
```

#### Scalability

Whilst GPs represent a powerful approach to nonlinear regression, they do have some limitations. GPs do not scale well with the number of observations, and standard GP approaches are not suitable when we have a very large datasets (thousands of observations). To overcome these limitations, approximate approaches to inference with GPs have been developed. 


Exercise 3.3: Write a function for determining differential expression for two genes. Hint: we are interested in comparing two models, and using Bayes' Factor to determine if the genes are differentially expressed.  
#### Advanced application 1: differential expression of time series {#application-1}

Differential expression analysis is concerned with identifying *if* two sets of data are significantly different from one another. For example, if we measured the expression level of a gene in two different conditions (control versus treatment), you could use an appropriate statistical test to determine whether the expression of that gene had been affected by the treatment. Most statistical tests used for this are not appropriate when dealing with time series data (illustrated in Figure \@ref(fig:timeser)). 

<div class="figure" style="text-align: center">
<img src="images/TimeSeries.jpg" alt="Differential expression analysis for time series. Here we have two time series with very different behaviour (right). However, as a whole the mean and variance of the time series is identical (left) and the datasets are not differentially expressed using a t-test (p&lt;0.9901)" width="55%" />
<p class="caption">(\#fig:timeser)Differential expression analysis for time series. Here we have two time series with very different behaviour (right). However, as a whole the mean and variance of the time series is identical (left) and the datasets are not differentially expressed using a t-test (p<0.9901)</p>
</div>

Gaussian processes regression represents a useful way of modelling time series, and can therefore be used as a basis for detecting differential expression in time series. To do so we write down two competing modes: (i) the two time series are differentially expressed, and are therefore best described by two independent GPs; (ii) the two time series are noisy observations from an identical underlying process, and are therefore best described by a single joint GP applied to the union of the data. 

Exercise 3.2 (optional): Write a function for determining differential expression for two genes. Hint: you will need to fit $3$ GPs: one to the mock/control, one to the infected dataset, and one to the union of mock/control and infected. You can use the Bayes' Factor to determine if the gene is differentially expressed.


```r
f <- data.frame(x=D[25:nrow(D),1]/48, y=D[25:nrow(D),geneindex])
par <- seq(.1,10,by=0.1)
ML <- matrix(rep(0, length(par)^2), nrow=length(par), ncol=length(par))
for(i in 1:length(par)) {
  for(j in 1:length(par)) {
    ML[i,j] <- calcMLn(f,par[i],par[j],0.05)
  }
}
persp3D(z = ML,theta = 120)
ind<-which(ML==max(ML), arr.ind=TRUE)
```

Now let's calculate the BF.


```r
lmap <- par[ind[1]]
varmap <- par[ind[2]]

f1 <- data.frame(x=D[1:24,1]/48, y=D[1:24,geneindex])
f2 <- data.frame(x=D[25:nrow(D),1]/48, y=D[25:nrow(D),geneindex])
f3 <- data.frame(x=D[,1]/48, y=D[,geneindex])

MLs <- matrix(rep(0, 3, nrow=3))
MLs[1] <- calcMLn(f1,lmap,varmap,0.05)
MLs[2] <- calcMLn(f2,lmap,varmap,0.05)
MLs[3] <- calcMLn(f3,lmap,varmap,0.05)

BF <- (MLs[1]+MLs[2]) -MLs[3]
BF
```

So from the Bayes' Factor there's some slight evidence for model 1 (differential expression) over model 2 (non-differential expression).

