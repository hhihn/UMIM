"""
Tiny ImageNet: Input Pipeline
Written by Patrick Coady (pcoady@alum.mit.edu)

Reads in jpegs, distorts images (flips, translations, hue and
saturation) and builds QueueRunners to keep the GPU well-fed. Uses
specific directory and file naming structure from data download
link below.

Also builds dictionary between label integer and human-readable
class names.

Get data here:
https://tiny-imagenet.herokuapp.com/
"""
import glob
import re
import tensorflow as tf
import random
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_filenames_labels(mode, target_classes=None):
  """Gets filenames and labels

  Args:
    mode: 'train' or 'val'
      (Directory structure and file naming different for
      train and val datasets)

  Returns:
    list of tuples: (jpeg filename with path, label)
  """
  label_dict, class_description = build_label_dicts(target_classes=target_classes)
  filenames_labels = []
  if mode == 'train':
    filenames = glob.glob('./tiny-imagenet-200/train/*/images/*.JPEG')
    for filename in filenames:
      match = re.search(r'n\d+', filename)
      try:
        label = str(label_dict[match.group()])
        filenames_labels.append((filename, label))
      except Exception:
        pass
  elif mode == 'val':
    with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
      for line in f.readlines():
        split_line = line.split('\t')
        filename = './tiny-imagenet-200/val/images/' + split_line[0]
        try:
          label = str(label_dict[split_line[1]])
          filenames_labels.append((filename, label))
        except Exception:
          pass

  return filenames_labels


def build_label_dicts(target_classes=None):
  """Build look-up dictionaries for class label, and class description

  Class labels are 0 to 199 in the same order as
    tiny-imagenet-200/wnids.txt. Class text descriptions are from
    tiny-imagenet-200/words.txt

  Returns:
    tuple of dicts
      label_dict:
        keys = synset (e.g. "n01944390")
        values = class integer {0 .. 199}
      class_desc:
        keys = class integer {0 .. 199}
        values = text description from words.txt
  """
  label_dict, class_description = {}, {}
  with open('./tiny-imagenet-200/wnids.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset = line[:-1]  # remove \n
      if target_classes is None or i in target_classes:
        label_dict[synset] = i
  with open('./tiny-imagenet-200/words.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset, desc = line.split('\t')
      desc = desc[:-1]  # remove \n
      if synset in label_dict:
        class_description[label_dict[synset]] = desc

  return label_dict, class_description


def read_image(item):
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
  filename = item[0]
  label = item[1]
  file = tf.io.read_file(filename)
  img = tf.image.decode_jpeg(file, channels=3)
  label = tf.strings.to_number(label, tf.int32)
  label = tf.cast(label, tf.uint8)

  return [img, label]


def batch_q(mode, config):
  """Return batch of images using filename Queue

  Args:
    mode: 'train' or 'val'
    config: training configuration object

  Returns:
    imgs: tf.uint8 tensor [batch_size, height, width, channels]
    labels: tf.uint8 tensor [batch_size,]

  """
  filenames_labels = load_filenames_labels(mode)
  random.shuffle(filenames_labels)
  filename_q = tf.train.input_producer(filenames_labels,
                                       num_epochs=config.num_epochs,
                                       shuffle=True)

  # 2 read_image threads to keep batch_join queue full:
  return tf.train.batch_join([read_image(filename_q, mode) for i in range(2)],
                             config.batch_size, shapes=[(56, 56, 3), ()],
                             capacity=2048)


def extract_classes_imagenet(classes, mode="train", num_classes=200):
  filenames_labels = load_filenames_labels(mode=mode, target_classes=classes)
  x, y = [], []
  for fl in filenames_labels:
    img, l = read_image(fl)
    img = tf.cast(img, dtype="float32") / 255.0
    x.append(img)
    y.append(l)
  print("\033[94m Loaded %d images for classes" % len(x), classes, "\033[0m")
  y = to_categorical(np.array(y), num_classes=num_classes)
  return x, y