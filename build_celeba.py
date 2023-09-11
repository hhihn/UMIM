import tensorflow as tf
import json
import numpy as np
from tensorflow.keras.utils import to_categorical

test_meta_path = "celeb_data/test/all_data_iid_01_05_keep_5_test_9.json"
train_meta_path = "celeb_data/train/all_data_iid_01_05_keep_5_train_9.json"
file_path = "celeb_data/celeba_resized/"


def read_image(filename):
    """Load next jpeg file from filename / label queue
  Randomly applies distortions if mode == 'train' (including a
  random crop to [56, 56, 3]). Standardizes all images.

  Args:
    filename_q: Queue with 2 columns: filename string and label string.
     filename string is relative path to jpeg file. label string is text-
     formatted integer between '0' and '199'
    mode: 'train' or 'val'

  Returns:
    [img, label]:
      img = tf.uint8 tensor [height, width, channels]  (see tf.image.decode.jpeg())
      label = tf.unit8 target class label: {0 .. 199}
  """
    file = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(file, channels=3)
    return img


def load_filenames_labels(mode="train", target_classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
    path = test_meta_path if mode == "test" else train_meta_path
    data = json.load(open(path))
    user_data = data['user_data']
    user_ids = list(user_data.keys())
    file_names = []
    labels =[]
    for uid in user_ids:
        if int(uid) in target_classes:
            fns = user_data[uid]['x']
            for f in fns:
                file_names.append(file_path+f)
            ys = user_data[uid]['y']
            labels.extend(ys)
    print("collected %d filenames and %d labels" % (len(file_names), len(labels)))
    return file_names, labels


def extract_classes_celeba(classes, mode="train", num_classes=2):
    filenames, labels = load_filenames_labels(mode=mode, target_classes=classes)
    x, y = [], []
    for f, l in zip(filenames, labels):
        img = read_image(f)
        img = tf.cast(img, dtype="float32") / 255.0
        x.append(img)
        y.append(l)
    print("\033[94m Loaded %d images for classes" % len(x), classes, "\033[0m")
    y = to_categorical(np.array(y), num_classes=num_classes)
    return x, y


