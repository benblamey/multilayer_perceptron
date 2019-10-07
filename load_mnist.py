import struct
import numpy as np

TRAINING_LABELS = 'mnist/t10k-labels-idx1-ubyte'
TEST_LABELS = 'mnist/train-labels-idx1-ubyte'

TEST_IMAGES = 'mnist/t10k-images-idx3-ubyte'
TRAINING_IMAGES = 'mnist/train-images-idx3-ubyte'


# See: http://yann.lecun.com/exdb/mnist/


def _read_labels(filename):
    with open(filename, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert (magic == 0x00000801)
        num_items = struct.unpack(">I", f.read(4))[0]
        print(f'There are {num_items} labels')

        labels = np.empty((num_items))

        for i in range(num_items):
            labels[i] = (f.read(1)[0])

        return labels


def _read_images(filename):
    with open(filename, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert (magic == 0x00000803)
        num_items = struct.unpack(">I", f.read(4))[0]
        print(f'There are {num_items} images')

        num_rows = struct.unpack(">I", f.read(4))[0]
        num_cols = struct.unpack(">I", f.read(4))[0]
        assert (num_rows == 28)
        assert (num_cols == 28)

        images = np.empty((num_items, num_rows, num_cols))

        for image_index in range(num_items):

            for row_index in range(num_rows):
                for col_index in range(num_cols):
                    images[image_index, row_index, col_index] = f.read(1)[0]

        return images


def get_training_labels():
    return _read_labels(TRAINING_LABELS)


def get_training_images():
    return _read_images(TRAINING_IMAGES)


def get_test_labels():
    return _read_labels(TEST_LABELS)


def get_test_images():
    return _read_images(TEST_IMAGES)


if __name__ == '__main__':
    # training_labels = read_labels(TRAINING_LABELS)
    # test_labels = read_labels(TEST_LABELS)
    # print(training_labels.size)
    # print(test_labels.size)

    images = _read_images(TEST_IMAGES)
    import matplotlib.pyplot as plt

    plt.imshow(images[0])
    plt.show()
    plt.imshow(images[1])
    plt.show()
    plt.imshow(images[2])
    plt.show()
    plt.imshow(images[3])
    plt.show()
