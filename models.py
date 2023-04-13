### code base: ai.berkeley.edu

import nn

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningrate = .1 # set learning rate to .1
        self.weight1 = nn.Parameter(784, 256) # weights and biases parameters set
        self.bias1 = nn.Parameter(1, 256)
        self.weight2 = nn.Parameter(256, 128)
        self.bias2 = nn.Parameter(1, 128)
        self.weight3 = nn.Parameter(128, 64)
        self.bias3 = nn.Parameter(1, 64)
        self.weight4 = nn.Parameter(64, 10)
        self.bias4 = nn.Parameter(1, 10)
        self.parameters = [self.weight1, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3, self.weight4, self.bias4]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        firstlayer = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.bias1)) # Rectified linear unit w/ adding bias vectors
        secondlayer = nn.ReLU(nn.AddBias(nn.Linear(firstlayer, self.weight2), self.bias2))
        thirdlayer = nn.ReLU(nn.AddBias(nn.Linear(secondlayer, self.weight3), self.bias3))
        outputlayer = nn.AddBias(nn.Linear(thirdlayer, self.weight4), self.bias4)
        return outputlayer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y) # computes batched softmax loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        sizeBatch = 100 # batch size
        loss = float('inf')
        validationAccurracy = 0
        while validationAccurracy < .98: # acc is under 98%
            for x, y in dataset.iterate_once(sizeBatch):
                loss = self.get_loss(x, y)
                gradients = nn.gradients(loss, self.parameters) # computes gradients of loss in respect to parameters
                loss = nn.as_scalar(loss) # as_scalar used to determine when to stop training
                for i in range(len(self.parameters)):
                    self.parameters[i].update(gradients[i], -self.learningrate) 
            validationAccurracy = dataset.get_validation_accuracy() # returns accuract of model on validation set

