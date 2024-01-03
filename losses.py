import tensorflow as tf
import tensorflow.keras.backend as K
import config

def SSIM_loss_graph(target, pred):    
    ssim_loss_weight = 0.4 #0.85
    mse_loss_weight = 0.6
    maxDepthVal = 176.0/1.0 #175.72, 184.0

    # Structural similarity (SSIM) index
    ssim_loss = tf.reduce_mean(
        1
        - tf.image.ssim(
            target, pred, max_val=maxDepthVal, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2
        ))
    
    # Point-wise depth
    mse = tf.keras.losses.MeanSquaredError()
    mse_loss = mse(target, pred)

    loss = (
        (ssim_loss_weight * ssim_loss)
        + (mse_loss_weight * mse_loss)
    )
    return loss

def depth_loss_function(y_true, y_pred, config):   
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
  
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    return (ssim_loss_weight * l_ssim) + (edge_loss_weight * K.mean(l_edges)) + (K.mean(l_depth))

def mrcnn_mask_edge_loss_graph(y_pred, y_true, edge_filters, smoothing_predictions, smoothing_gt, norm, weight_entropy, weight_factor):
    """
    Extra loss to enforce the edges which are present in the groundtruth mask
    :param y_pred: Predicted mask shape [rois, width, height], with values from 0 to 1
    :param y_true: Groundtruth mask shape [rois, width, height], with values from 0 to 1
    :param edge_filters: List of edge detectors to use
    :param smoothing_predictions: Use a Gaussian smoothing on the predictions before calculating the edges
    :param smoothing_gt: Use a Gaussian smoothing on the groundtruth before calculating the edges
    :param norm: Type of the norm to calculate the Edge Loss. Supported values: lp for p in [1..5]
    :param softmax_entropy_weight: Use the edge loss to weight an additional cross entropy
            loss instead of using it as a L^p norm
    :return: loss
    """
    # sobel kernels
    sobel_x_kernel = tf.reshape(tf.constant([[1, 2, 1],
                                             [0, 0, 0],
                                             [-1, -2, -1]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='sobel_x_kernel')
    sobel_y_kernel = tf.reshape(tf.constant([[1, 0, -1],
                                             [2, 0, -2],
                                             [1, 0, -1]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='sobel_y_kernel')

    # prewitt kernels
    prewitt_x_kernel = tf.reshape(tf.constant([[1, 0, -1],
                                             [1, 0, -1],
                                             [1, 0, -1]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='prewitt_x_kernel')
    prewitt_y_kernel = tf.reshape(tf.constant([[1, 1, 1],
                                             [0, 0, 0],
                                             [-1, -1, -1]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='prewitt_y_kernel')

    # prewitt kernels
    kayyali_senw_kernel = tf.reshape(tf.constant([[6, 0, -6],
                                             [0, 0, -0],
                                             [-6, 0, 6]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='kayyali_senw_kernel')
    kayyali_nesw_kernel = tf.reshape(tf.constant([[-6, 0, 6],
                                             [0, 0, 0],
                                             [6, 0, -6]], dtype=tf.float32),
                                shape=[3, 3, 1, 1], name='kayyali_nesw_kernel')

    # roberts kernels
    roberts_x_kernel = tf.reshape(tf.constant([[1, 0],
                                              [0, -1]], dtype=tf.float32),
                                shape=[2, 2, 1, 1], name='roberts_x_kernel')
    roberts_y_kernel = tf.reshape(tf.constant([[0, -1],
                                               [1, 0]], dtype=tf.float32),
                                shape=[2, 2, 1, 1], name='roberts_y_kernel')

    # laplace kernel
    laplacian_kernel = tf.reshape(tf.constant([[1, 1, 1],
                                               [1, -8, 1],
                                               [1, 1, 1]], dtype=tf.float32),
                                  shape=[3, 3, 1, 1], name='laplacian_kernel')

    gaussian_kernel = tf.reshape(tf.constant([[0.077847, 0.123317, 0.077847],
                                              [0.123317, 0.195346, 0.1233179],
                                              [0.077847, 0.123317, 0.077847]], dtype=tf.float32),
                                 shape=[3, 3, 1, 1], name='gaussian_kernel')

    filter_map = {
        "sobel-x": sobel_x_kernel,
        "sobel-y": sobel_y_kernel,
        "roberts-x": roberts_x_kernel,
        "roberts-y": roberts_y_kernel,
        "prewitt-x": prewitt_x_kernel,
        "prewitt-y": prewitt_y_kernel,
        "kayyali-senw": kayyali_senw_kernel,
        "kayyali-nesw": kayyali_nesw_kernel,        
        "laplace": laplacian_kernel
    }

    lp_norm_map = {
        "l1": 1,
        "l2": 2,
        "l3": 3,
        "l4": 4,
        "l5": 5
    }

    if norm not in lp_norm_map:
        raise ValueError("The `norm` '{0}' is not supported. Supported values are: [l1...l5]".format(norm))

    edge_filters = tf.concat([filter_map[x] for x in edge_filters], axis=-1)

    # Add one channel to masks
    y_pred = tf.expand_dims(y_pred, -1, name='y_pred')
    y_true = tf.expand_dims(y_true, -1, name='y_true')

    if smoothing_predictions:
        # First filter with gaussian to smooth edges of predictions
        y_pred = tf.nn.conv2d(input=y_pred, filters=gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')

    y_pred_edges = tf.nn.conv2d(input=y_pred, filters=edge_filters, strides=[1, 1, 1, 1], padding='SAME')

    if smoothing_gt:
        # First filter with gaussian to smooth edges of groundtruth
        y_true = tf.nn.conv2d(input=y_true, filters=gaussian_kernel, strides=[1, 1, 1, 1], padding='SAME')
    y_true_edges = tf.nn.conv2d(input=y_true, filters=edge_filters, strides=[1, 1, 1, 1], padding='SAME')

    def append_magnitude(edges, name=None):
        magnitude = tf.expand_dims(tf.sqrt(edges[:, :, :, 0] ** 2 + edges[:, :, :, 1] ** 2), axis=-1)
        return tf.concat([edges, magnitude], axis=-1, name=name)

    def lp_loss(y_true, y_pred, p):
        return K.mean(K.pow(K.abs(y_pred - y_true), p), axis=-1)

    def smoothness_loss(y_true, y_pred, p):
        weight_smoothness = K.exp(-K.abs(y_true))
        smoothness = y_pred * weight_smoothness
        smoothness = smoothness[:, :, :, 0] + smoothness[:, :, :, 1]
        return K.mean(K.pow(K.abs(smoothness), p))

    # calculate the edge agreement loss per pixel
    pixel_wise_edge_loss = K.switch(tf.size(y_true_edges) > 0,
                                    lp_loss(y_true=y_true_edges, y_pred=y_pred_edges, p=lp_norm_map[norm]),
                                    tf.constant(0.0))

    # multiply the pixelwise edge agreement loss with a scalar factor
    pixel_wise_edge_loss = weight_factor * pixel_wise_edge_loss

    if weight_entropy:
        pixel_wise_cross_entropy_loss = K.switch(tf.size(y_true) > 0,
                                                 K.binary_crossentropy(target=y_true, output=y_pred),
                                                 tf.constant(0.0))

        weighted_cross_entropy_loss = tf.squeeze(pixel_wise_cross_entropy_loss) * tf.exp(pixel_wise_edge_loss / 16)
        weighted_cross_entropy_loss = K.mean(weighted_cross_entropy_loss) / 1

        return weighted_cross_entropy_loss
    else:
        # return the mean of the pixelwise edge agreement loss
        edge_loss = K.mean(pixel_wise_edge_loss)
        return edge_loss