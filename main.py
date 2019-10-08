# https://davidstutz.de/recognizing-handwritten-digits-mnist-dataset-twolayer-perceptron/
# https://github.com/davidstutz/matlab-mnist-two-layer-perceptron/blob/master/trainStochasticSquaredErrorTwoLayerPerceptron.m

# Load data
import numpy as np
import load_mnist
import logit

# LEARNING_RATE = 0.1 in his code
LEARNING_RATE = 0.05

test_labels = load_mnist.get_test_labels()
test_images = load_mnist.get_test_images() / 255

training_labels = load_mnist.get_training_labels()
training_images = load_mnist.get_training_images() / 255

assert training_labels.shape[0] == training_images.shape[0]

num_training_images = training_images.shape[0]


# Convert classes to be 1-hot

def labels_to_hot(labels: np.ndarray):
    hot = np.zeros((labels.shape[0], 10), dtype=int)
    for i, label in enumerate(labels):
        hot[i, label] = 1
    return hot


test_labels_hot = labels_to_hot(test_labels)
training_labels_hot = labels_to_hot(training_labels)

# Create initial weights

NUM_EPOCHS = 90
BATCH_SIZE = 1000
NUM_HIDDEN_UNITS = 100

hidden_weights = np.random.random((NUM_HIDDEN_UNITS, (28 * 28)))
output_weights = np.random.random((10, NUM_HIDDEN_UNITS))

hidden_weights = (hidden_weights.transpose() / hidden_weights.sum(axis=1)).transpose()
output_weights = (output_weights.transpose() / output_weights.sum(axis=1)).transpose()

errors = []

# 'epoch' means 'time period', i.e. in this case 'set of iterations'
for epoch_index in range(NUM_EPOCHS):
    print(f'epoch {epoch_index}')

    # we do online/stotastic/sequential gradient descent.
    # this just means we update the weights using one input vector (chosen at random, with replacement)
    #  at a time.
    # this handles redundancy well in the dataset.
    for batch_index in range(BATCH_SIZE):
        # choose an input vector (image)
        image_index = np.random.randint(num_training_images)
        input_vector = np.reshape(training_images[image_index], 28 * 28)

        # forward propagate
        hidden_input = np.sum(hidden_weights * input_vector, axis=1)
        hidden_output = logit.logit(hidden_input)

        final_input = np.sum(output_weights * hidden_output, axis=1)
        final_output = logit.logit(final_input)

        target = training_labels_hot[image_index]

        # back propagate

        outputDelta = logit.dlogit_by_dx(final_input) * (final_output - target)
        hiddenDelta = logit.dlogit_by_dx(hidden_input) * (outputDelta * output_weights.transpose()).transpose()

        output_weights = output_weights - LEARNING_RATE * np.outer(outputDelta, hidden_output)
        hidden_weights = hidden_weights - LEARNING_RATE * np.outer(hiddenDelta.sum(axis=0), input_vector)

        # hidden_weights = hidden_weights - hidden_weights.sum(axis=1)/hidden_weights.shape[1]
        # output_weights = output_weights - output_weights.sum(axis=1)/hidden_weights.shape[1]
        #
        # hidden_weights = np.where(hidden_weights < -10, -10, hidden_weights)
        # hidden_weights = np.where(hidden_weights > 10, 10, hidden_weights)
        #
        # output_weights = np.where(output_weights < -10, -10, output_weights)
        # output_weights = np.where(output_weights > 10, 10, output_weights)
        #
        # hidden_weights = (hidden_weights.transpose() / hidden_weights.sum(axis=1)).transpose()
        # output_weights = (output_weights.transpose() / output_weights.sum(axis=1)).transpose()


    def classify(input_vector):
        classify_input_hidden_layer = (input_vector * hidden_weights).sum(axis=1)
        classify_output_hidden_layer = logit.logit(classify_input_hidden_layer)
        classify_input_final_layer = (classify_output_hidden_layer * output_weights).sum(axis=1)
        classify_output_final_layer = logit.logit(classify_input_final_layer)

        return classify_output_final_layer


    # error
    error = 0
    for i in range(BATCH_SIZE):
        input_vector = training_images[i].reshape(28 * 28)
        target_vector = training_labels_hot[i]

        target_guess = classify(input_vector)

        error = error + np.linalg.norm(target_guess - target_vector)

    error = error / BATCH_SIZE
    errors.append(error)

import matplotlib.pyplot as plt

plt.plot(errors)
plt.show()

# Classify test data using those weights
# ...gives a vector of results

# ??? Backprop with derivatives?

# Normalize? (>> he doesn't seem to do it.)
