from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
from BayesianDenseMoe import *
from tensorflow.python.keras.utils import losses_utils
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import tensorflow_similarity as tfsim
import tensorflow_probability as tfp
from build_celeba import extract_classes_celeba
tfd = tfp.distributions
tf.compat.v1.enable_eager_execution()

import functools
import numpy as np
from global_settings import *
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

CROP_TO = 32


def build_dataset(x, y, batch_size, epochs):
    dataset = tf.data.Dataset.from_tensor_slices((tf.stack(x), tf.stack(y)))
    dataset = dataset.shuffle(buffer_size=len(x), seed=SEED)
    dataset = dataset.map(cifar_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat(epochs)
    return dataset


def flip_random_crop(image):
    # With random crops we also apply horizontal flipping.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    return image


def color_jitter(x, strength=[0.4, 0.4, 0.4, 0.1]):
    x = tf.image.random_brightness(x, max_delta=0.8 * strength[0])
    x = tf.image.random_contrast(
        x, lower=1 - 0.8 * strength[1], upper=1 + 0.8 * strength[1]
    )
    x = tf.image.random_saturation(
        x, lower=1 - 0.8 * strength[2], upper=1 + 0.8 * strength[2]
    )
    x = tf.image.random_hue(x, max_delta=0.2 * strength[3])
    # Affine transformations can disturb the natural range of
    # RGB images, hence this is needed.
    x = tf.clip_by_value(x, 0, 255)
    return x


def color_drop(x):
    x = tf.image.rgb_to_grayscale(x)
    x = tf.tile(x, [1, 1, 3])
    return x

def _gaussian_blur(image, padding='SAME'):
  """Blurs the given image with separable convolution.
  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.
  Returns:
    A Tensor representing the blurred image.
  """
  sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32, seed=SEED)
  kernel_size = 3
  radius = tf.cast(kernel_size / 2, dtype=tf.int32)
  kernel_size = radius * 2 + 1
  x = tf.cast(tf.range(-radius, radius + 1), dtype=tf.float32)
  blur_filter = tf.exp(-tf.pow(x, 2.0) /
                       (2.0 * tf.pow(tf.cast(sigma, dtype=tf.float32), 2.0)))
  blur_filter /= tf.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf.shape(image)[-1]
  blur_h = tf.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = image.shape.ndims == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf.expand_dims(image, axis=0)
  blurred = tf.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf.squeeze(blurred, axis=0)
  return blurred


def random_apply(p, x, func):
    """Randomly apply function func to x with probability p."""
    return tf.cond(
        tf.less(
            tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),#, seed=SEED),
            tf.cast(p, tf.float32),
        ),
        lambda: func(x),
        lambda: x,
    )

def _random_inverse_greyscale(image, random_func):
    uniform_random = random_func(shape=[], minval=0, maxval=1.0)
    mirror_cond = math_ops.less(uniform_random, .5)
    result = control_flow_ops.cond(
        mirror_cond,
        lambda: 1.0 - image,
        lambda: image)
    return result

def random_inverse_greyscale(image, seed=None):
  """Randomly invert greyscale image."""
  random_func = functools.partial(random_ops.random_uniform, seed=SEED)
  return _random_inverse_greyscale(image, random_func)

def _random_noise(image, random_func):
    mask = tf.random.uniform(shape=tf.shape(image), minval=0.0, maxval=1.0)
    mean = 0.0
    std = 0.001
    mask = tf.cast(mask, dtype=image.dtype) * std + mean
    uniform_random = random_func(shape=[], minval=0, maxval=1.0)
    mirror_cond = math_ops.less(uniform_random, .5)
    result = control_flow_ops.cond(
        mirror_cond,
        lambda: image + mask,
        lambda: image)
    return result

def random_noise(image, seed=None):
    """Randomly add salt and pepper noise"""
    random_func = functools.partial(random_ops.random_uniform, seed=SEED)
    return _random_noise(image, random_func)

def _random_gaussian_blur(image, random_func):
    uniform_random = random_func(shape=[], minval=0, maxval=1.0)
    mirror_cond = math_ops.less(uniform_random, .5)
    result = control_flow_ops.cond(
        mirror_cond,
        lambda: _gaussian_blur(image),
        lambda: image)
    return result

def random_gaussian_blur(image, seed=None):
    """Randomly add salt and pepper noise"""
    random_func = functools.partial(random_ops.random_uniform, seed=SEED)
    return _random_noise(image, random_func)

def _random_translation(image, random_func):
    uniform_random = random_func(shape=[], minval=0, maxval=1.0)
    mirror_cond = math_ops.less(uniform_random, .5)
    result = control_flow_ops.cond(
        mirror_cond,
        lambda: tf.image.rot90(image),
        lambda: image)
    return result

def random_translation(image, seed=None):
    random_func = functools.partial(random_ops.random_uniform, seed=SEED)
    return _random_translation(image, random_func)

def _rot90(x):
    return tf.image.rot90(x)

# random color jitter
def _jitter_transform(x):
    return tfsim.augmenters.augmentation_utils.color_jitter.color_jitter_rand(
        x,
        np.random.uniform(0.0, 0.4),
        np.random.uniform(0.0, 0.4),
        np.random.uniform(0.0, 0.4),
        np.random.uniform(0.0, 0.1),
        "multiplicative",
    )

def _crop_and_resize(x):
    """
    Crops and resizes image by getting a random square of size ranging from 75% of the width to 100% of width.
    This cropped portion is then resized to the original height and width, which may cause stretching and
    distortion for rectangular images.
    """
    height = tf.cast(tf.shape(x)[0], dtype="float32")
    width = tf.cast(tf.shape(x)[1], dtype="float32")
    min_cropsize_multiplier = 0.75
    max_cropsize_multiplier = 1.0
    rand_width = tf.random.uniform(
        shape=[],
        minval=int(min_cropsize_multiplier * width),
        maxval=int(max_cropsize_multiplier * width),
        dtype=tf.int32,
        seed=SEED,
    )

    crop = tf.image.random_crop(x, (rand_width, rand_width, 3), seed=SEED)
    crop_resize = tf.image.resize(crop, (height, width))
    return crop_resize

# # random grayscale
def _grayscale_transform(x):
    return tfsim.augmenters.augmentation_utils.color_jitter.to_grayscale(x)
#asdf
def cifar_simsiam_augmenter(img, blur=True, area_range=(0.3, 1.0), eval=False, seed=1234):
    """SimSiam augmenter.

    The SimSiam augmentations are based on the SimCLR augmentations, but have
    some important differences.
    * The crop area lower bound is 20% instead of 8%.
    * The color jitter and grayscale are applied separately instead of together.
    * The color jitter ranges are much smaller.
    * Blur is not applied for the cifar10 dataset.

    args:
        img: Single image tensor of shape (H, W, C)
        blur: If true, apply blur. Should be disabled for cifar10.
        area_range: The upper and lower bound of the random crop percentage.

    returns:
        A single image tensor of shape (H, W, C) with values between 0.0 and 1.0.
    """
    # random resize and crop. Increase the size before we crop.
    # random horizontal flip
    img = tf.image.random_flip_left_right(img)#, seed=SEED)
    img = tf.image.random_flip_up_down(img)#, seed=SEED)
    img = random_apply(func=_rot90, p=0.5, x=img)
    if eval:
        img = random_apply(func=_rot90, p=0.25, x=img)
        img = random_apply(func=_rot90, p=0.125, x=img)
    img = random_apply(func=_jitter_transform, p=0.8, x=img)
    img = random_apply(func=_grayscale_transform, p=0.2, x=img)
    if eval:
        img = random_apply(func=_gaussian_blur, p=0.5, x=img)
    img = random_apply(func=_crop_and_resize, p=0.5, x=img)
    return img, 0


# @tf.function
def simsiam_augmenter(img, neg_imgs, area_range=(0.4, 1.0), neg=False, eval=False):
    """SimSiam augmenter.

    The SimSiam augmentations are based on the SimCLR augmentations, but have
    some important differences.
    * The crop area lower bound is 20% instead of 8%.
    * The color jitter and grayscale are applied separately instead of together.
    * The color jitter ranges are much smaller.
    * Blur is not applied for the cifar10 dataset.

    args:
        img: Single image tensor of shape (H, W, C)
        blur: If true, apply blur. Should be disabled for cifar10.
        area_range: The upper and lower bound of the random crop percentage.

    returns:
        A single image tensor of shape (H, W, C) with values between 0.0 and 1.0.
    """
    # and tf.random.uniform(shape=[], minval=0.0, maxval=1.0) <= 0.5
    img = tf.reshape(img, shape=(28, 28, 1))
    if neg:
        if len(neg_imgs):
            idx = tf.random.uniform(shape=(), minval=0, maxval=len(neg_imgs), dtype=tf.int32)
            in_img = neg_imgs[idx]
            t = 0.0
            return in_img, 0.0
        else:
            in_img = tf.random.uniform(minval=0.0, maxval=1.0, shape=tf.shape(img))
            return in_img, 0.0
    else:
        in_img = img
        t = 1.0

    img = tf.image.random_flip_left_right(in_img)
    img = random_translation(img)
    img = random_gaussian_blur(img)
    img = tfsim.augmenters.augmentation_utils.cropping.crop_and_resize(
            img, 28, 28, area_range=area_range
    )

    img /= tf.reduce_max(img)

    return img, t

def process_wrapper(neg_imgs):
    @tf.function()
    def process(img, y):
        view1, _ = simsiam_augmenter(img, neg_imgs=neg_imgs)
        view2, t = simsiam_augmenter(img, neg_imgs=neg_imgs)
        img = tf.reshape(img, shape=(784, 1))
        view1 = tf.reshape(view1, shape=(784, 1))
        if view2 is not None:
            view2 = tf.reshape(view2, shape=(784, 1))
        return (img, view1, view2, 0, y)
    return process

@tf.function()
def cifar_process(img, y):
    view1, _ = cifar_simsiam_augmenter(img)
    view2, t = cifar_simsiam_augmenter(img)
    view3, t = cifar_simsiam_augmenter(img, eval=True)
    return (img, view1, view2, view3, y)


def extract_classes_mixed(classes, num_classes, x, y, mode="test"):
    ds, target_classes = classes
    if ds == "cifar":
        return extract_classes(classes=target_classes, x=x, y=y, num_classes=num_classes)
    elif ds == "celeb":
        return extract_classes_celeba(classes=target_classes, num_classes=num_classes, mode=mode)
    return None

def extract_classes(classes, x, y, num_classes=10, frac=1.0):
    sub_x = []
    sub_y = []
    for i, c in enumerate(classes):
        classes_idx = np.where(y == c)[0]
        sub_x.append(x[classes_idx])
        sub_y.extend(y[classes_idx])
    x = np.vstack(sub_x)
    y = to_categorical(np.array(sub_y), num_classes=num_classes)
    if frac < 1.0:
        num_idx = int(frac*np.shape(x)[0])
        sub_idx = np.random.choice(a=np.arange(0, np.shape(x)[0], 1), size=num_idx, replace=False)
        x = x[sub_idx]
        y = y[sub_idx]
    return x, y


def eval_experts(models, dispatched_inputs, dispatched_outputs):
    task_acc = 0.0
    total_samples = 0
    num_models = len(models)
    embeddings = []
    for expert_inputs, expert_outputs, model, mi in zip(dispatched_inputs, dispatched_outputs, models, range(num_models)):
        if len(expert_inputs):
            y_pred, latent_embedding, mean_prediction_entropy = models[mi].latent_call(expert_inputs, training=False)
            embeddings.append(latent_embedding)
            y_test_pred = np.argmax(y_pred, axis=-1)
            task_acc += accuracy_score(y_true=tf.argmax(expert_outputs, axis=-1), y_pred=y_test_pred) * len(y_pred)
            total_samples += len(y_pred)
    return task_acc / total_samples, embeddings


@tf.function(experimental_relax_shapes=True, jit_compile=True)
def gaussian_kernel(x, width=0.15, aspiration=0.85):
    return tf.exp(-(tf.subtract(aspiration, x)))


def dispatch(models, x, y, xxp=None, xxn=None, xxx=None, eval=False, first_run=False, max_num_models=5,
             similarity_threshold=0.75, new_model_cooldown=0, task_id=0):
    model_loss_per_input = []
    large_num = 1e9
    all_samples = tf.concat([x, xxp, xxn], axis=0)
    const_class_accs = [0.0, 0.0, 0.0]
    if len(models) > 1 or not first_run or eval:
        for mi, model in enumerate(models):
            #if eval or mi == len(models) - 1:
            _, embeddings, _ = model.latent_call(all_samples, training=not eval)
            proj_latent_anch, proj_latent_pos, proj_latent_neg = tf.split(embeddings, 3, 0)
            contrastive_loss = model.total_similarity(proj_latent_anch, proj_latent_pos, proj_latent_neg)
            if eval:
                logits_pos = tf.matmul(proj_latent_anch, proj_latent_pos, transpose_b=True) / model.temperature# + tf.matmul(hidden2, hidden1, transpose_b=True)
                logits_neg = tf.matmul(proj_latent_anch, proj_latent_neg, transpose_b=True) / model.temperature# + tf.matmul(hidden2, hidden1, transpose_b=True)
                # logits_neg_2 = tf.matmul(proj_latent_anch, proj_latent_neg_2, transpose_b=True) / model.temperature# + tf.matmul(hidden2, hidden1, transpose_b=True)
                for li, logits in enumerate([logits_pos, logits_neg]):
                    samples = tf.shape(logits)[0]
                    y_pred = tf.cast(tf.argmax(logits), dtype="int32")
                    y_true = tf.cast(tf.range(samples), dtype="int32")
                    const_class_accs[li] += (tf.reduce_sum(tf.cast(tf.math.equal(y_pred, y_true), dtype="int32")) / samples) / len(models)
            #else:
             #   contrastive_loss = tf.zeros(shape=(tf.shape(x)[0], 1)) - large_num
            model_loss_per_input.append(contrastive_loss)
        possible_gates = len(models)
        model_loss_per_input = tf.concat(model_loss_per_input, axis=-1)
        if not eval and not first_run and len(models) < max_num_models and new_model_cooldown <= 0:
            model_loss_per_input = tf.concat([model_loss_per_input, tf.zeros((tf.shape(model_loss_per_input)[0],1))+large_num*2],
                                             axis=-1)
            possible_gates += 1
        gates, top_k_logits = top_k_gating(model_loss_per_input, soft=False)
        #if not eval:
        val, idx, count = tf.unique_with_counts(tf.argmax(gates, axis=-1))
        winner_idx = tf.argmax(count)
        winner = val[winner_idx]
        gates = tf.zeros_like(gates, dtype="int64") + winner
        gates = tf.one_hot(gates[:, 0], depth=possible_gates)
        dispatcher = SparseDispatcher(possible_gates, gates)
        dispatched_idxs = []
        dispatched_x = dispatcher.dispatch(x)

        return dispatched_x, \
                dispatcher.dispatch(xxp) if xxp is not None else dispatched_x, \
                dispatcher.dispatch(xxn) if xxn is not None else dispatched_x, \
                dispatcher.dispatch(xxx) if xxx is not None else dispatched_x, \
                dispatcher.dispatch(y), \
                dispatched_idxs, \
                tf.reduce_max(top_k_logits, axis=-1), \
                const_class_accs,
    else:
        return [x], [xxp], [xxn], [xxx], [y], [tf.zeros(shape=(len(x), 1))+0.9], tf.zeros(shape=(len(x), 1))+0.99, [0.0, 0.0]


def _my_top_k(x, k, soft=True):
    """GPU-compatible version of top-k that works for very small constant k.
    Calls argmax repeatedly.
    tf.nn.top_k is implemented for GPU, but the gradient, sparse_to_dense,
    seems not to be, so if we use tf.nn.top_k, then both the top_k and its
    gradient go on cpu.  Once this is not an issue, this function becomes
    obsolete and should be replaced by tf.nn.top_k.
    Args:
    x: a 2d Tensor.
    k: a small integer.
    soft: sample or use argmax
    Returns:
    values: a Tensor of shape [batch_size, k]
    indices: a int32 Tensor of shape [batch_size, k]
    """
    if k > 10:
        return tf.math.top_k(x, k)
    values = []
    indices = []
    depth = tf.shape(x)[1]
    for i in range(k):
        if not soft:
            idx = tf.argmax(x, 1)
            values.append(tf.reduce_max(x, 1))
        else:
            dist = tfd.Categorical(logits=x)
            idx = dist.sample()
            values.append(dist.log_prob(idx))
        indices.append(idx)
        if i + 1 < k:
            x += tf.one_hot(idx, depth, -1e9)
    return tf.stack(values, axis=1), tf.cast(tf.stack(indices, axis=1), dtype=tf.int32)


def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


def _rowwise_unsorted_segment_sum(values, indices, n):
    """UnsortedSegmentSum on each row.
    Args:
    values: a `Tensor` with shape `[batch_size, k]`.
    indices: an integer `Tensor` with shape `[batch_size, k]`.
    n: an integer.
    Returns:
    A `Tensor` with the same type as `values` and shape `[batch_size, n]`.
    """
    batch, k = tf.unstack(tf.shape(indices), num=2)
    indices_flat = tf.reshape(indices, [-1]) + tf.cast(tf.divide(tf.range(batch * k), k), dtype=tf.int32) * n
    ret_flat = tf.math.unsorted_segment_sum(tf.reshape(values, [-1]), indices_flat, batch * n)
    return tf.reshape(ret_flat, [batch, n])


def top_k_gating(gating_logits, soft=True):
    k = 1
    top_logits, top_indices = _my_top_k(gating_logits, k, soft)
    top_k_logits = tf.slice(top_logits, [0, 0], [-1, k])
    top_k_indices = tf.slice(top_indices, [0, 0], [-1, k])
    top_k_gates = tf.nn.softmax(top_k_logits)
    gates = _rowwise_unsorted_segment_sum(top_k_gates, top_k_indices, tf.shape(gating_logits)[-1])

    return gates, top_k_logits


def get_idx_name(name, postflag):
    post = ""
    if postflag:
        post = "post_"
    idx = -1
    # kernels
    if post+"bias" in name:
        idx = 12
    elif post+"expert_embedding_bias" in name:
        idx = 13
    elif post+"2_expert_embedding_bias" in name:
        idx = 14
    elif post+"q_mu_log_var" in name:
        idx = 15
    elif post+"q_tau_rate" in name:
        idx = 16
    elif post+"components_q_mu_kernels" in name:
        idx = 17
    elif post+"components_q_tau_kernels" in name:
        idx = 18
    elif "gamma" in name:
        idx = 19
    elif "beta" in name:
        idx = 20
    elif "moving_mean" in name:
        idx = 21
    elif "moving_variance" in name:
        idx = 22
    
    return idx


def copy_weights(old_model, new_model, verbose=False, final_copy=True):
    if old_model is None or new_model is None:
        print("\033[91m WARNING: one of the models is None \033[0m")
    if old_model is not None:
        if verbose:
            print("copy weights %s to %s" % (old_model.expert_name, new_model.expert_name))
        for new_layer, old_layer in zip(new_model.model_layers.layers, old_model.model_layers.layers):
            old_weights = old_layer.get_weights()
            new_weights = new_layer.get_weights()
            old_weight_obj = old_layer.weights
            new_weight_obj = new_layer.weights
            prior_weights = [[] for _ in range(100)]
            prior_weight_names = [[] for _ in range(100)]
            for wobj, weight in zip(old_weight_obj, old_weights):
                idx = get_idx_name(wobj.name, postflag=True)
                if idx >= 0:
                    prior_weights[idx] = weight
                    prior_weight_names[idx] = wobj.name
                    if verbose:
                        print("old model: detected", prior_weight_names[idx], "and saved on slot", idx)
                elif "prior" not in wobj.name:
                    print("old model: -> did not recognize", wobj.name)
                elif "post" in wobj.name:
                    print("\033[91m WARNING: could not find entry of posterior weight %s  \033[0m" % wobj.name)
            updated_new_weights = []

            for wobj, weight, old_wobj, old_weight in zip(new_weight_obj, new_weights, old_weight_obj,
                                                          old_weights):
                idx = get_idx_name(wobj.name, postflag=False)

                if idx >= 0:
                    updated_new_weights.append(prior_weights[idx])
                    if verbose:
                        print("new model: detected", wobj.name, "and overwriting using", prior_weight_names[idx])
                else:
                    updated_new_weights.append(old_weight)
                    if "prior" in wobj.name:
                        print("\033[91m WARNING: could not find weights for prior %s  \033[0m" % wobj.name)
                    if verbose:
                        print("new model: could not find %s and copied weights of %s" % (wobj.name, old_wobj.name))
            if len(updated_new_weights):
                new_layer.set_weights(updated_new_weights)

def task_mi_from_mat(mat):

    q = np.mean(mat, axis=0) + 1e-3
    task_mi = 0.0
    for p in mat:
        task_mi = task_mi + np.sum(p * np.log(p / q))
    return (task_mi / mat.shape[0]) / np.log(2.0)


def bwt_from_mat(mat):
    avg_bwt = 0.0
    bwt = 0.0
    N = mat.shape[1]
    mean_acc = np.mean(mat[:, N - 1])
    for i in range(mat.shape[0]):
        a_0 = mat[i, i]
        bwt = bwt + (mat[i, N - 1] - a_0)
        for j in range(i, N):
            avg_bwt = avg_bwt + (mat[i, j] - a_0)
    return mean_acc, avg_bwt / ((N * (N - 1)) * 2.0), bwt / (N - 1)


def uniformity(embeddings_per_model):
    print("embs per model", len(embeddings_per_model))
    uniformities = []
    # for ei in range(len(embeddings_per_model)):
    #     print("embeddings", len(embeddings_per_model[ei]))
    for embeddings in embeddings_per_model:
        if len(embeddings):
            u = []
            sub_idx = np.random.choice(a=np.arange(0, len(embeddings), 1), size=min(500, len(embeddings)))
            embeddings = np.array(embeddings)[sub_idx]
            for e1 in range(len(embeddings)):
                print("uniformities: %d/%d" % (len(u), (len(embeddings)*(len(embeddings)-1)/2)), end="\r")
                for e2 in range(e1+1, len(embeddings)):
                    u.append(np.linalg.norm(embeddings[e1] - embeddings[e2], ord=2))
            u = np.mean(u)
            uniformities.append(u)
    return uniformities


def pairwise_distance(embeddings_per_model):
    uniformities = []
    sim_loss = tf.keras.losses.MeanSquaredError(
        reduction=losses_utils.ReductionV2.NONE,
        name='cosine_similarity'
    )
    for e1, embeddings in enumerate(embeddings_per_model):
        if len(embeddings):
            for e2 in range(e1+1, len(embeddings_per_model)):
                dists = sim_loss(embeddings, embeddings_per_model[e2])
                uniformities.append(np.mean(dists))
    return uniformities


def alignment(x, y, models, num_classes=10):
    alignments = []
    if len(models) < 2:
        return [0.0]
    for ci in range(num_classes):
        m_embs = [[] for _ in range(len(models))]
        xi, yi = extract_classes(x=x, y=y, classes=[ci])
        for mi, m in enumerate(models):
            _, m_emb, _ = m.latent_call(xi, training=False)
            m_embs[mi].extend(m_emb)
        unif = np.mean(pairwise_distance(m_embs))
        alignments.append(unif)
    return alignments

