"""Train and Eval for BReG-NeXt on AffectNet database (categorical model).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tflearn
import os.path
import sys
import time
import numpy as np
import keras
from keras import backend as K
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
INPUT_FILE='../tfrecords/training_FER2013_sample.tfrecords'
validation_file = '../tfrecords/validation_FER2013_sample.tfrecords'

Snapshots_path="Snapshots/"
att_number = len([i for i in os.listdir(Snapshots_path)]) + 1
Snapshots_path='Snapshots/categorical_attempt_' + str(att_number) + '/'
if not os.path.exists(Snapshots_path):
    os.makedirs(Snapshots_path)
Snapshots_path += 'checkpoints'
Logs_path="Logs/categorical_attempt_" + str(att_number) + '/'
if not os.path.exists(Logs_path):
    os.makedirs(Logs_path)


initial_learning_rate = 0.0001
n_classes = 8

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 9

def residual_block(incoming, nb_blocks, out_channels, downsample=False,
                   downsample_strides=2, activation='relu', batch_norm=True,
                   bias=True, weights_init='variance_scaling',
                   bias_init='zeros', regularizer='L2', weight_decay=0.0001,
                   trainable=True, restore=True, reuse=False, scope=None,
                   name="ResidualBlock"):
    """ Residual Block.
    A residual block as described in MSRA's Deep Residual Network paper.
    Full pre-activation architecture is used here.
    Input:
        4-D Tensor [batch, height, width, in_channels].
    Output:
        4-D Tensor [batch, new height, new width, nb_filter].
    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        nb_blocks: `int`. Number of layer blocks.
        out_channels: `int`. The number of convolutional filters of the
            convolution layers.
        downsample: `bool`. If True, apply downsampling using
            'downsample_strides' for strides.
        downsample_strides: `int`. The strides to use when downsampling.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'linear'.
        batch_norm: `bool`. If True, apply batch normalization.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (see tflearn.initializations) Default: 'uniform_scaling'.
        bias_init: `str` (name) or `tf.Tensor`. Bias initialization.
            (see tflearn.initializations) Default: 'zeros'.
        regularizer: `str` (name) or `Tensor`. Add a regularizer to this
            layer weights (see tflearn.regularizers). Default: None.
        weight_decay: `float`. Regularizer decay parameter. Default: 0.001.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: A name for this layer (optional). Default: 'ShallowBottleneck'.
    References:
        - Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.
        - Identity Mappings in Deep Residual Networks. Kaiming He, Xiangyu
            Zhang, Shaoqing Ren, Jian Sun. 2015.
    Links:
        - [http://arxiv.org/pdf/1512.03385v1.pdf]
            (http://arxiv.org/pdf/1512.03385v1.pdf)
        - [Identity Mappings in Deep Residual Networks]
            (https://arxiv.org/pdf/1603.05027v2.pdf)
    """
    resnet = incoming
    in_channels = incoming.get_shape().as_list()[-1]

    # Variable Scope fix for older TF
    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:

        name = scope.name #TODO

        for i in range(nb_blocks):

            identity = resnet

            if not downsample:
                downsample_strides = 1

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = tflearn.conv_2d(resnet, out_channels, 3,
                             downsample_strides, 'same', 'linear',
                             bias, weights_init, bias_init,
                             regularizer, weight_decay, trainable,
                             restore)

            if batch_norm:
                resnet = tflearn.batch_normalization(resnet)
            resnet = tflearn.activation(resnet, activation)

            resnet = tflearn.conv_2d(resnet, out_channels, 3, 1, 'same',
                             'linear', bias, weights_init,
                             bias_init, regularizer, weight_decay,
                             trainable, restore)


            with tf.name_scope('shortcut_mod'):
              multiplier = tf.Variable(1, dtype=tf.float32, trainable=True, name='a')
              multiplier2 = tf.Variable(1, dtype=tf.float32, trainable=True, name='c')
              with tf.name_scope('shortcut_mod_function'): 
                identity = tf.div(tf.atan(tf.div(tf.multiply(multiplier,identity),tf.sqrt(tf.add(tf.pow(multiplier2,2),1)))),tf.multiply(multiplier,tf.sqrt(tf.add(tf.pow(multiplier2,2),1))))

            if downsample_strides > 1:
                identity = tflearn.avg_pool_2d(identity, downsample_strides,
                                               downsample_strides)
            
            # Projection to new dimension
            if in_channels != out_channels:
                ch = (out_channels - in_channels)//2
                identity = tf.pad(identity,
                                  [[0, 0], [0, 0], [0, 0], [ch, ch]])
                in_channels = out_channels
            resnet = resnet + identity

    return resnet

def focal_loss2(y_true, y_pred,gamma=2., alpha=.25):
  """ focal loss implementation with Keras.
  """
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
  return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def BReG_NeXt(_X):
  """BReG_NeXt implementation. Returns feature map before softmax.
  """
  net = tflearn.conv_2d(_X, 32, 3, regularizer='L2', weight_decay=0.0001)
  net = residual_block(net, 7, 32,activation='elu')
  net = residual_block(net, 1, 64, downsample=True,activation='elu')
  net = residual_block(net, 8, 64,activation='elu')
  net = residual_block(net, 1, 128, downsample=True,activation='elu')
  net = residual_block(net, 7, 128,activation='elu')
  net = tflearn.batch_normalization(net)
  net = tflearn.activation(net, 'elu')
  net = tflearn.global_avg_pool(net)
  # Regression
  logits = tflearn.fully_connected(net, n_classes, activation='linear')
  return logits

def decode(serialized_example):
  """Parses an image and label from the given `serialized_example`."""
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image=tf.reshape(image, [64,64,3])
  image.set_shape([64,64,3])

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)
  label_categorical = tf.one_hot(
    label,
    depth= n_classes,
    on_value=1,
    off_value=0,
    dtype=tf.int32,
  )
  label_categorical = tf.reshape(label_categorical, [n_classes])
  label_categorical.set_shape([n_classes])

  return image, label_categorical

def augment(image, label):
  """Placeholder for data augmentation."""

  def process(img): 
    img = tf.image.random_flip_left_right(img)

    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.7, 1.3)


    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img2):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img2], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(64, 64))
        crops = tf.cast(crops,tf.uint8)

        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    img = tf.cond(choice < 0.5, lambda: img, lambda: random_crop(img))
    return img

  # Only apply augmeting 25% of the time
  choice2 = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
  image = tf.cond(choice2 <= 0.5, lambda: image, lambda: process(image))

  return image, label


def normalize(image, label):
  """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
  image = tf.cast(image, tf.float32) * (1. / 255) #- 0.5

  image = tf.stack([tf.subtract(image[:,:,0] , 0.5727663) ,
                    tf.subtract(image[:,:,1] , 0.44812188),
                    tf.subtract(image[:,:,2] , 0.39362228)], axis = 2 )

  return image, label

def clip(image, label):
  image = tf.clip_by_value(tf.cast(image, tf.float32), 0, 255)
  return image,label

def validation_inputs(batch_size, train = False,  num_epochs = 1):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, NUM_CLASSES).

    This function creates a one_shot_iterator, meaning that it will only iterate
    over the dataset once. On the other hand there is no special initialization
    required.
  """
  if not num_epochs:
    num_epochs = None
    
  filename = validation_file

  with tf.name_scope('input'):
    # TFRecordDataset opens a binary file and reads one record at a time.
    # `filename` could also be a list of filenames, which will be read in order.
    dataset = tf.data.TFRecordDataset(filename)

    # The map transformation takes a function and applies it to every element
    # of the dataset.
    dataset = dataset.map(decode, num_parallel_calls = 4)
    dataset = dataset.map(clip, num_parallel_calls = 4)
    dataset = dataset.map(normalize, num_parallel_calls = 4)

    # The shuffle transformation uses a finite-sized buffer to shuffle elements
    # in memory. The parameter is the number of elements in the buffer. For
    # completely uniform shuffling, set the parameter to be the same as the
    # number of elements in the dataset.
    dataset = dataset.shuffle(1000 + 3 * batch_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()

def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, NUM_CLASSES).

    This function creates a one_shot_iterator, meaning that it will only iterate
    over the dataset once. On the other hand there is no special initialization
    required.
  """
  if not num_epochs:
    num_epochs = None

  filename=INPUT_FILE

  with tf.name_scope('input'):
    # TFRecordDataset opens a binary file and reads one record at a time.
    # `filename` could also be a list of filenames, which will be read in order.
    dataset = tf.data.TFRecordDataset(filename)

    # The map transformation takes a function and applies it to every element
    # of the dataset.
    dataset = dataset.map(decode, num_parallel_calls = 4)
    dataset = dataset.map(augment, num_parallel_calls = 4)
    dataset = dataset.map(clip, num_parallel_calls = 4)
    dataset = dataset.map(normalize, num_parallel_calls = 4)

    # The shuffle transformation uses a finite-sized buffer to shuffle elements
    # in memory. The parameter is the number of elements in the buffer. For
    # completely uniform shuffling, set the parameter to be the same as the
    # number of elements in the dataset.
    dataset = dataset.shuffle(1000 + 3 * batch_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()


def run_training():
  """Train model for a number of steps."""
  
  image_batch, label_batch = inputs(
        train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

  image_batch_val, label_batch_val = validation_inputs(
        train=False, batch_size=FLAGS.batch_size, num_epochs = 1)

  image_batch_placeholder = tf.placeholder(tf.float32, shape=(None, 64, 64, 3),name="image_batch_placeholder")
  label_tensor_placeholder = tf.placeholder(tf.int32, shape=(None,n_classes),name="label_tensor_placeholder")

  # Build a Graph that computes predictions from the inference model.
  pred = BReG_NeXt(image_batch_placeholder)

  # Define loss and optimizer
  with tf.name_scope('loss'): 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label_tensor_placeholder))
    # for focal loss use:
    # cost = focal_loss2(y_true = label_tensor_placeholder, y_pred = tf.nn.softmax(pred))

  with tf.name_scope('optimizer'):
    batch = tf.Variable(0, trainable=False)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate_op = tf.train.exponential_decay(
        initial_learning_rate,                # Base learning rate.
        batch,  # Current index into the dataset.
        8000,          # Decay step.
        0.80,                # Decay rate.
        staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_op).minimize(cost,global_step=batch)

  # Evaluate model
  with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(label_tensor_placeholder,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  tf.summary.scalar ('accuracy',accuracy)
  tf.summary.scalar ('loss',cost)

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.summary.merge_all()
  saver = tf.train.Saver(max_to_keep=10)

  # The op for initializing the variables.
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

  # Create a session for running operations in the Graph.
  with tf.Session() as sess:
    # Initialize the variables (the trained variables and the
    # epoch counter).
    sess.run(init_op)
    summary_writer = tf.summary.FileWriter(Logs_path, sess.graph)
    
    try:
      step = 0
      best_val_acc = 0.0
    
      while True:  # Train until OutOfRangeError
        start_time = time.time()

        # Run one step of the model.  The return values are
        # the activations from the `train_op` (which is
        # discarded) and the `loss` op.  To inspect the values
        # of your ops or variables, you may include them in
        # the list passed to sess.run() and the value tensors
        # will be returned in the tuple from the call.
        image_out, label_out = sess.run([image_batch, label_batch])
        _,loss_val,accuracy_val,summary_str,lr_value = sess.run([optimizer, cost,accuracy,summary_op,learning_rate_op],feed_dict={image_batch_placeholder: image_out, label_tensor_placeholder: label_out})

        duration = time.time() - start_time

        # Print an overview fairly often.
        if step % 100 == 0:
          summary_writer.add_summary(summary_str, step)
          

          with tf.Session() as sess2:
            result_cum = np.array([], dtype=np.int64).reshape(0,2)
            try:
              while True:
                image_out_val, label_out_val = sess2.run([image_batch_val, label_batch_val])
                pred_val, validation_accuracy = sess.run([pred,accuracy],feed_dict={image_batch_placeholder: image_out_val, label_tensor_placeholder: label_out_val})

                shape1 = label_out_val.argmax(axis=1).shape[0]
                shape2 = pred_val.argmax(axis=1).shape[0]
                result = np.concatenate((label_out_val.argmax(axis=1).reshape(shape1,1),pred_val.argmax(axis=1).reshape(shape2,1)),axis= 1)

                result_cum = np.concatenate((result_cum,result))
            except tf.errors.OutOfRangeError:
              validation_acc = accuracy_score(result_cum[:,0], result_cum[:,1])
              print('Step %d: loss = %.5f acc = %.4f lr = %.8f validation acc = %.4f' % (step, loss_val, accuracy_val, lr_value, validation_acc))

              if validation_acc > best_val_acc :
                saver.save(sess, Snapshots_path,  global_step=step)
                print('Best so far. Model saved.')

              best_val_acc = max(best_val_acc,validation_acc)

        step += 1
    except tf.errors.OutOfRangeError:
      print('Done training for %d epochs, %d steps. Best validation accuracy: %.4f' % (FLAGS.num_epochs,step,best_val_acc))


def main(_):
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=100,
      help='Number of epochs to run trainer.')
  parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
  parser.add_argument(
      '--train_dir',
      type=str,
      default='./',
      help='Directory with the training data.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
