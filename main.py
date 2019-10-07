# https://davidstutz.de/recognizing-handwritten-digits-mnist-dataset-twolayer-perceptron/
# https://github.com/davidstutz/matlab-mnist-two-layer-perceptron/blob/master/trainStochasticSquaredErrorTwoLayerPerceptron.m

# Load data
import numpy as np
import load_mnist

test_images = load_mnist.get_test_images()
training_images = load_mnist.get_training_images()
test_labels = load_mnist.get_test_labels()
training_labels = load_mnist.get_training_labels()

num_training_images = training_images.shape[0]

# Convert classes to be 1-hot

def labels_to_hot(labels: np.ndarray):
    hot = np.zeros((labels.shape[0], 10), dtype=int)
    for i,label in enumerate(labels):
        hot[i, label] = 1
    return hot

test_labels_hot = labels_to_hot(test_labels)
training_labels_hot = labels_to_hot(training_labels)


# Create initial weights

NUM_EPOCHS = 100
BATCH_SIZE = 100


# 'epoch' means 'time period', i.e. in this case 'set of iterations'
for epoch_index in range(NUM_EPOCHS):

    # we do online/stotastic/sequential gradient descent.
    # this just means we update the weights using one input vector (chosen at random, with replacement)
    #  at a time.
    # this handles redundancy well in the dataset.
    for batch_index in range(BATCH_SIZE):

        # choose an input vector.
        image_index = np.random.rand(num_training_images)



        # Classify test data using those weights
        # ...gives a vector of results

        # ??? Backprop with derivatives?

        # Normalize? (>> he doesn't seem to do it.)
