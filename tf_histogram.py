import tensorflow as tf

def get_histogram(tensor, nbins=10):
    # Computes histogram with equally spaced bins (assumes the input is normalized in the range 0-1)
    # INPUT : features tensor B x n x nfeatures
    # OUTPUT : histogram tensor B x nfeatures x nbins

    tensor_length = len(tensor.get_shape().as_list())
    if tensor_length < 3 :
        row_dim = 0
    else:
        row_dim = 1

    nrows = tensor.get_shape().as_list()[row_dim]
    one_hot_bins = tf.one_hot(tf.cast(tf.floor(tensor*nbins), tf.int32),
                                    depth = nbins, on_value=1.0, off_value=0.0)
    hist = tf.reduce_sum(one_hot_bins, axis=row_dim)/nrows

    return hist

if(__name__ == '__main__'):

    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    import numpy as np

    points = tf.random_uniform([32, 100, 100]) # batch tensor example
    #points = tf.constant([[0.08, 0.23, 0.34, 0.45],[0.38, 0.35, 0.89, 0.78]]) # single tensor example
    nbins = 10
    hist_tensor = get_histogram(points, nbins)
    sess = tf.InteractiveSession()
    hist_values = hist_tensor.eval()

    # Display the histogram as a bar plot

    display_histogram = hist_values[0][:]
    # the histogram of the data
    rects = plt.bar(np.arange(0,nbins,1),display_histogram , 1, color="blue")


    plt.xlabel('bins')
    plt.ylabel('Probability')
    plt.title(r'Tensor Values Histogram')
    plt.axis([0-0.5, nbins-0.5, 0, 1])
    plt.grid(True)

    plt.show()