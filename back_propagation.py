"""
8/2018 nt:
578hw1.py (modified from NN578_network.py)
==============

Modified from the NNDL book code "network.py" to be 
compatible with Python 3.

Also from "network2.py", the function ("load(filename)") is
added to this file, renamed as "load_network(filename)".
This function loads a saved network encoded in a json file.

"""

"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import json

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]




    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        else: n_test =0
        
        n = len(training_data)
        result_lst=np.array([])
        train_output_lst=[]
        test_output_lst=[]
        for epoch in range(epochs): #xrange(epochs):
            
            random.shuffle(training_data)
            #print("len(training_data)", len(training_data))
            mini_batches = [training_data[k:k+mini_batch_size]for k in range(0, n, mini_batch_size)]
            #print("mini_batches",mini_batches)

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            #Call Function:evaluate() to test on training data after backpropagation
            #Store all the training measurment result into a list
            train_output=self.evaluate(training_data)
            train_output_lst.append(list(train_output))
            
            #Only for Demonstration: Print Testing Set Result for each Epoch
            #print("[epoch %d]"%(epoch), end=" ")
            #print ("Training: MSE=%0.7f, CrossEntropy=%0.7f, LogLikelihood=%0.7f, Correct=%d/%d , Accuracy=%0.4f" % (
              #  train_output[2], train_output[3],train_output[4],train_output[0], n, train_output[1]))

            #Only for Demonstration: Print Testing Set Result for each Epoch
            if test_data:
                #Call Function:evaluate() to test on training data after backpropagation
                test_output = self.evaluate(test_data)
                test_output_lst.append(list(test_output))
               # print ("          Test:     MSE=%0.7f, CrossEntropy=%0.7f, LogLikelihood=%0.7f, Correct=%d/%d , Accuracy=%0.4f" % ( 
                #    test_output[2],test_output[3],test_output[4],test_output[0], n_test, test_output[1]))
            else: 
                #print ("          Test:     No testing data") #Print if NO testing data pass in 
                #Store all the training measurment result into a list
                test_output_lst.append(list(np.array([])))

            #Break Learning Training Loop when Accuracy = 100%
            if  int(train_output[1]) == 1: 
                break

        return [train_output_lst, test_output_lst]

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #Initialize list to hold the activations (hidden layer outputs),inputs and outputs
        activations=list(np.empty((self.num_layers,1)))
        for i in range(len(self.sizes)):
            activations[i]= np.zeros((self.sizes[i],1))
        
        # feedforward
        activation = x
        activations[0] = x  # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for i,(b, w) in enumerate (zip(self.biases, self.weights)):
            #print("b",b)
            #print("w",w)
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations[i+1] = activation
        
        #Check the collected activations
        #print("activations", activations)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        
        #Number of test instances passing into this function
        n = len(test_data)
        
        test_result_lst = [(self.feedforward(x), y) for (x, y) in test_data]
        
        #Find maximum indexes from the predicted outputs and true outputs List
        test_correct_case_lst = [(np.argmax(a), np.argmax(y)) for (a, y) in test_result_lst]
                
        #Calculate the number cases predicted vs true outputs
        correct_case = sum([int(x == y) for (x, y) in test_correct_case_lst])

        #Calculate Accuracy of predicted outputs
        accuracy = [correct_case / n]

        #Calculate MSE for Predicted vs True outputs
        test_mse_lst = [np.mean(np.array([(0.5)*(np.linalg.norm(a-y)**2) for (a, y) in test_result_lst]))]

        #Calculate Cross-Entropy for Predicted vs True outputs
        test_crossEntropy_lst = [np.mean(np.array([np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))) for (a, y) in test_result_lst]))]

        #Calculate Log-Likelihood for Predicted vs True outputs
        test_logLikelihood_lst = [np.mean(np.array([-np.sum(np.nan_to_num(np.log(a))) for (a, y) in test_result_lst]))]
            
        #Combine all the Error meausrment metrics into a List format.
        outLst=[correct_case] + accuracy  + test_mse_lst+test_crossEntropy_lst+test_logLikelihood_lst
        
        return( outLst)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#### Savinging a Network to a json file
def save_network(net, filename):
    """Save the neural network to the file ``filename``."""
    data = {"sizes": net.sizes,
            "weights": [w.tolist() for w in net.weights],
            "biases": [b.tolist() for b in net.biases]#,
            #"cost": str(net.cost.__name__)
           }
    f = open(filename, "w")
    json.dump(data, f)
    f.close()
        
#### Loading a Network from a json file
def load_network(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    #cost = getattr(sys.modules[__name__], data["cost"])
    #net = Network(data["sizes"], cost=cost)
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorize_target(n, target):
    """Return an array of shape (n,1) with a 1.0 in the target position
    and zeroes elsewhere.  The parameter target is assumed to be
    an array of size 1, and the 0th item is the target position (1).
    """
    e = np.zeros((n, 1))
    e[int(target[0])] = 1.0
    return e    
