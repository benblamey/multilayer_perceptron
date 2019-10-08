import struct
import numpy as np

TRAINING_LABELS = 'mnist/t10k-labels-idx1-ubyte'
TRAINING_IMAGES = 'mnist/t10k-images-idx3-ubyte'

TEST_IMAGES = 'mnist/train-images-idx3-ubyte'
TEST_LABELS = 'mnist/train-labels-idx1-ubyte'

# See: http://yann.lecun.com/exdb/mnist/

def _read_labels(filename):
    with open(filename, "rb", 1024) as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert (magic == 0x00000801)

        num_items = struct.unpack(">I", f.read(4))[0]
        print(f'There are {num_items} labels')

        labels = np.fromfile(f, dtype=np.dtype('u1'), count=num_items, offset=0)

    return labels


def _read_images(filename):
    with open(filename, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert (magic == 0x00000803)

        num_images = struct.unpack(">I", f.read(4))[0]
        print(f'There are {num_images} images')

        num_rows = struct.unpack(">I", f.read(4))[0]
        num_cols = struct.unpack(">I", f.read(4))[0]
        assert (num_rows == 28)
        assert (num_cols == 28)

        images = np.reshape(np.fromfile(f,
                                        dtype=np.dtype('u1'),
                                        count=num_images * num_rows * num_cols,
                                        offset=0),
                            (num_images, num_rows, num_cols))

        return images / 255


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

    plt.imshow(images[0], cmap=plt.cm.binary)
    plt.show()
    plt.imshow(images[1])
    plt.show()
    plt.imshow(images[2])
    plt.show()
    plt.imshow(images[3])
    plt.show()
