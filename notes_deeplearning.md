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


##Activation functions for general architectures
-The choise of what activation function depends on the kind of data (sparve vs dense)
-These functions are applied on the hidden or output layers. Normally the input layer is raw
-A more continuous distribution of input data is best modeled by ReLU activation fuction

###Output layer decisions
-Depending on the target of the output, the choice of activation function changes
-It can be: Regression, binary classification, multi-class classification

####Regression
-Objective: Get real numbers
-Use linear activation function.

####Binary Classification
-Objective: Get a 0.0~1.0 representing probability
-Use Sigmoid activation functions

####Multi-class Classification
-Objective1: Get the class with the biggest score
-Use Softmax  with argmax()
-Objective2: Get the scores for all classes to get a multi class system (person+car+dog for example)
-Use Sigmoid givin 0.0~1.0 probability for each class

##Loss Function
-Quantify the agreement between the predicted output and the expected output
-Squared Loss, Logistic Loss, Hinge Loss, Negative Log Likelihood 
-Also falling in the categories classifation, regression and reconstruction.

###Recontruction Cross Entropy
-First apply a gaussian noise then punishes results not similar to the input without the noise
-This ensures that it will try to learn  new features
-Used on Restricted Boltzmann Machines

##Optimazation
-First order: Calculates a Jacobian matrix
--Matrix with the partial derivates of loss funcion values with respect to each parameter
-Second order: Calculates the derivate of the Jacobian matrix by approaching Hessian
-Second order, because they also take into account the relation between parameters, take better steps, but take longer to calculate

###First order methods
-The stochastic gradient descent is several orders of magnitude faster than batch gradient descent
--Its faster because uses noise as well when calculating the gradient, making the convergion faster

###Second order methods
-Describe the curvature at each point of the jacobian
-Converge in fewer steps, but takes longer to calculate a step

####L-BFGS (Limited memory BFGS)
-Does not compute the whole Hessian matrix to save space in memory
-Works faster because uses aproximated second order information
-L-BFGS and conjugate gradient descent are faster than Stochastic Gradient Descent.

####Conjugate Gradient
-Focuses on minimizing the conjugate L2 norm
-Like normal gradient descent, but requires that each step after the first to be conjugate of its previous

####Hessian-Free
-Like the newton method but minimizes the quadratic function of that method faster.
-Its a powerful method adapted to NN in 2010
-Uses Conjugate Gradient to find the minimum of the quadratic

##Hyperparameters
-Any parameter that influenciates the performance 
-Keep in mind that some parameters are incompatible with each other

###Layer size
-Number of neurons on a given layer... duh
-Input and output layers are defined by the problem. With output being 1(for regression) or the number of classes
-The number of neurons on hidden layers is directly related to how complex the problem is
--Carefull thou as a network too big might take forever to train or ends up overfitting

###Magnitude
####Learning rate
-A big one helps converge faster, but overshoot the minimum
-Too slow and it will take forever to train
-The best solution is a dynamic learning rate
####Momentum
-SGD dont use it as default and because of that has a chance of erratic steps, like a 0 gradient or one too big
-Some commom techniques to regulate it: momentum, RMStop, Adam, AdaDelta
-Momentum is a factor between 0.0 ~ 1.0
####AdaGrad
-Adaptevely (got it?) uses subgradient methods to adjust the learning rate
-It speeds up at the beginning and slows it down when close to the minima decreasing error.
-AdaGrad is the square root of the sum of squares of a window of the most recent gradient computations.
####AdaDelta
-Like AdaGrad but keeping only the most recent history instead of all the gradients
####Adam
-Gets a learning rate by estimating the first and second moments of the gradient 

###Regularization
-Measures taken against overfitting
-"Cause it to overfit, then regularizate the hell out of it" - Geoffery Hinton
-Dropout, DropConnect: Mutes parts of the input so that the network has to learn new positions
####Dropout
-Ommit(deactivate) a hidden unit(neuron) at random during training.
-Speeds up training
-Whem ommited, the neuron is not used in the forward nor backpropagation
####Dropconnect
-Same as dropout, but instead of dropping a whole neuron, drops a connection between them
####L1
-Prevent parameters from getting too big compared to all the others
-Has automatic feature selection
-Considered inefficient on dense space since provide sparce outputs
-Uses multiplication by absolute value rather than squared ones. This makes some weights fall to 0 and some getting big
-But make it easier to interpret the weights
####L2
-More computationally efficient than L1. Also has non-sparce output
-Does NOT have automatic feature selection
-Decreses the squared weights
-Multiply half the sum of the squared weights by a weight-cost parameter
-Helps ignore weights it cannot use and smooths the output

###Mini-Batching
-Sends batches of data to be trained instead of 1 entry only
-Used to improve training speed by making better use of hardware.(GPU)

#Building Blocks of a Deep Network
##Unsupervised Layer-wise pretraining
-Useful when we have a relativelly big amount of unlabeled data 
-Uses Restricted Boltsman Machine to pretrain the first layers using the input, and then the other layers using the output of the previous one
-Even thou it creates an overhead, it helps initialize the main NN with beter weights.

###Restricted Boltzmann Machine
-Probabilistic Feed-forwartd NN model that uses 2 bias instead of 1
-Good for finding features and dimensionality reduction
-The neurons never connect to other neurons of the same layer
-Trained to be able to reconstruct the input data set

####Network Layout
-Composed of 5 parts: Visible/Hidden Units, Weights, Visible/Hidden Bias Unit
-Every visible unit is connected to every hidden unit and vice-versa, but they are not connected between themseves
-The hidden layer is responsable for the feature detection
-The visible layer can receive training vectors
-Each layer has a bias unit set to always ON
-Trained just like a normal network using activation functions
-Uses Contrastive Divergence to train. Minimizes the KL Divergence by samplying k steps from a markov chain

####Training
-By randomly picking the parameters for the reconstruction and measuring the KL distance, it behaves like a loss function that can be minimized

####Other uses of RBMs
-Dimensionality reduction
-Classification
-Regression
-Collaborative Filtering
-Topic modeling



###Auto-encoders
-Another Feed-forward NN that uses another bias to calculate the error of constructing the original input
-Its unsupervises since uses only the inputs for training, no need for label.
-Share similiarities with MLP in the sense that it also has input-hidden-output layers. But the output layer is the same size as the input
-Helps detect anomalies. Specially on systems where is difficult to define what is anomalous

####Common variants of auto-encoders
#####Compression auto-encoders
-They funnel down at the middle creating a bottleneck then increase again until the output.
-The diagram end up looking like an hourglass
#####Denoising auto-encoders
-Recieve a corrupted version of the input and has to learn how to create the uncorrupted version.


#Major architectures of Deep networks
##Unsupervised pre-trained Neural Network
-Comprehend networks like auto-encoders and Deep belief networks

###Deep Belief Networks
-It is made of layers of RBMs for pretraining followed bya normal feed-forward for fine tunning

####Feature extraction with RBM
-Use layers of RBMs to learn progressivily higher features of the data by putting the output of a RBM as the input of the next

####Initializing the feed-forward Neural Network
-The layers of features calculated by the RBMs initialize the weights of the feed-forward neural network
-Helps in guiding the network into a easier conversionable error-space zone

####Fine tuning phase of the feed-forward Neural Network
-Uses a smaller learning rate to produce a gentle backpropagation

##Convolutional Neural Networks
-Uses convolutions to learn higher order features from the input data
-Specially good in object recognition on images
-Capable of producing rotation/translation invariant features from raw data
-Also used on natural language generation and sentiment analysis
-Works best when the input data has repeating patterns that appear spacially close to each other. Like patches of pixels on the same objects
-CNN is good because normal feed-forward neural networks would need too much memory to train an image. A 32x32 rgb image would have 3072 weights only for one neuron of the hidden layer
-The convolutional layers can be arrange in 3d volumes (width height and depth). Think of them as analogous to an image width, height and depth(3 for rgb)

###CNN Architecture Overview
-Normally devided into: Input, Feature Extraction and Classification layers
-The input receives an 3d volume representing the image
-The feature extraction layers normally have a sequence of Convolutional -> RELu(activation function) -> Pooling , layers
-The classification layer is a fully connected network with 1 or more hidden layers that get the result of the Feature extration layer as input







