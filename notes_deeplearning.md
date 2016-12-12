take a look on Stuart Russel book, Artificial Inteligence: a modern approach 

Statistics:
-Use the cross validation as rotation with the splits. A normal number of splits 10.
 .create N folds
 .train with N-1 of the folds and test with the one left behind
 .shift the test fold to the next and retrain
 .keep doing it until all the folds have been a test at one time

-maximum likelyhood and minimum cost are about the same thing


###
#Evaluation

-Type I error: False positive
-Type II error: False negative

Sencibility (or recall): used to evauate well we avoid false negatives
  TP / (TP+FN)

Specificity shows how well a model avoids false positives
  TN / (TN+FP)

OBS: some models require tunning for one side or the other depending on the cost of missing. A medical model would 
want to minimize false negatives for a disease if the cost is dyeing.

Accuracy: Measures how close the model is to the true values in quantity
  (TP+TN)/(TP+TN+FP+FN)
  -can be misleading as it focus only on quantity. A event that has 90% chance of happining can easily be modeled with 90% accuracy
  by just making input == event

Precision: Measures how well repeated measurements over same conditions produce same answers
  TP/(TP+FP)

F1: good measure of accuracy when dealing with binary classification. also used as a good overall score of how well the model is performing as it combines recall and precision
  2TP/(2TP+FP+FN)

###
#Neural nets

-The simple perceptron can only separate datasets that are linearly separable
-Bias: They insure that at least tome neurons are activated per layer regardless of signal strenght gatanteeing that the model will try new interpretations to the signal
-Activation: is said to happen if a neuron's output is not zero (can be +or- thou)

#Feed forward neural net
-has: Single input layer, 1 or more fully conected hidden layers, single output layer
-Neurons of the same layer use the same activation function

#Activation functions

-Responsable for adding non-linearity to the network when applied to the hidden layers
-They are functions with S shape. Known as sigmoidal functions.

##Linear
-f(x) = Wx
-The identity function
-Used on single layer perceptrons. 
-For networks, it usually used on the input layer

##Sigmoid
-Reduce extreme values or outliers
-Convert values of any range into probabilities from 0 to 1. With most of the values on 0 or 1

##Tanh
-A trigonometric function analogous to the tangent
-Has a range of -1 to 1, which makes it easear to deal with negative values

###Hard Tanh
-Same as tanh, but with a clipping factor where everything outside the range is collapsed to it

##Softmax
-Analogous to the logistic regressions, but can be used to classify several classes instead of 2 only.
-Normally seen on the output layer
-Has a mutually exclusive aspect. Only one class will be selected
-In case of thousands of labels, a tree version exists. Hierarchical softmax.

##Rectfied Linear
-Only activates after the input achieves a certain level. after that it has a linear relation with the input
-The ReLU is the state of the art now, as it has proven to train better networks than the sigmoid

#Loss Functions
-Agregate the error into a single number representative of how well the neural network is doing. In other words, how close it is from the ideal network. 
-Transform the task of finding the perfect weights and bias into a optimization problem (reduction of the loss function)

##Loss functions for regression
###Mean Squared Loss, MSL
-h_wb = 1/N * sum_N( 1/M * sum_M( (yij_hat + yij)^2 ) )
-Used when we need real values on the regression output
-It optimizes for the mean, which means that it is a convex function and will always converto to a global minimum
  -might take forever thou
-As a downside, it is sensible to outliers
-To run from outliers it is wise to use median instead of mean

###Mean Absolute Error, MAE
-h_wb = 1/2N * sum_N( sum_M( |yij_hat-yij| ))
-Like MSL but averaging over the absolute error over the dataset

###Mean Squared Log Error, MSLE
-h_wb = 1/N sum_N( sum_M( (log yij_hat, log yij)^2 ))

###Mean Absolute Percentage Error, MAPE
-h_wb = 1/N sum_N( sum_M( 100 * |yij_hat-yij| / yij ))

-Less discriminatory based on range than the other loss functions

-Even thou MSLE and MAPE can handle big ranges, it is common practice to normalize the input before training and use MSL or MAE. optimizing for mean or median depending on the situation

##Loss functions for classification
-Can provide probablities to each class

###Hinge Loss
-h_wb = 1/N sum_N(max(0, 1 - yij * yij_hat))
-Most used loss function for when doing hard classification (0/1, for example)
-known to be a convex function
-Normally used for binary classification, but can be extended with *one vs all* or *one vs one*

###Logistic Loss
-Used when probabilities of each class is required
-Since all the probabilities has to sum to one, the very last layer of the network has to be a softmax
-Since sigmoids does not model the dependencies of outputs, cannot be used here. even thou they also give values between 0 and 1.
-It need to be trainined for all classes using maximum likelihood
-h_wb = mult_N(yi_hat^yi * (1-yi_hat)^(1-yi))
####Negative log likelihood
-to simplify calculations, use a log so that the multiplications become somation
-also negates the expression so it corresponds to a loss
-h_wb = -sum_N(yi * log(yi_hat) + (1-yi) * log(1-yi_hat))
-And for multiple classes: h_wb = -sum_N(sum_M(yij * log(yij_hat)))

##Loss functions for reconstruction
-Used in neural nets that has the function of recreating its input
-These neural nets can be used as auto encoders or even restricted Boltz-mann machines
-KL Divergence: Dkl(y||y_hat) = -sum_N(yi * log(yi/yi_hat))
-used in information theory as informatin divergence or information gain. Amount of information lost when using y_hat to aproximate y.

#Hyperparameters
##Learning rate
-Controls how much the error affects the weights.
-A bigger learning rate can train faster, but is prone to overshooting
-A small learning rate can converge better, but might take forever and can get stuck on local minimas
-A dynamic learning rate is the way to go

##Regularization
-Defined as coeficients L1 and L2
-Prevents some of the weight from getting too big, which prevents overfiting when the amount of data is still not big enough
-With enough data, no regularization is needed because the sheer size of the data set regulates the training

##Momentum
-Based on inertia, helps the gradient step to keeps its overall direction 
-Helps the learning not getting stuck on local minima
-It is to the learning rate what learning rate is to the weights

##Sparsity
-Recognizes that some important features does not appear that much
-The bias make sure the neuros responsable for features that dont appear that much are still being activated.  Making them stay arround "just in case"


############
#Defining Deep Networks
-Unsupervised Pre-Trainined Networks
-Convolutional Neural Networks
-Recurrent Neural Networks
-Recursive Neural Networks

##Deep Reinforcement learning
-Does not tell the network what actions to do
-The input is the information acquired from the envirioment
-The enforcement is positive if the actions are going toward the objective
-The difference from usual reinforcement learning is that the universal function aproximator is a NN. Even thou the proof of convergence dont work anymore, the results are good

##Advances on Network Architecture
-Deep Belief Networks make use of Restricted Boltzmann Machines for pretraing features
-Use hybrids of convolutional and recurrent neuro nets for classifying videos

###Automatic feature extraction
-The different architectures extract features differently. And some are better for different types of features
-The features normally start as something crude as edges and is refined layer by layer until having something that resambles the object

##Generative modeling
-Using machine learning to create data. Not just interpret it

###Inceptionalism
-Use a NN with its layers in reverse order together with an input image with a prior constraint.
-This enhances parts of the image based on the classification, as hallucinating.
-For example, parts that look like a fish will be enhanced with a fish texture and so on

###Modeling Artistic Style
-Have a model training with a artist style then use an image as input to be transformed into it
-Like having a family photo painted by Monet
-The network extract the style into its parameters

###Generative Adversarial Network
-Generate novel images by using the distribution of data seen by the network
-Used for mixing faces

###Recurrent Neural Networks
-Can generate coherent sequence of characters by being feed 3 or more
-Generates Shakespeare phrases after being trained by its works





