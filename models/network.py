import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
import tf_util
from dgcnn_util import get_sampled_edgeconv, get_edgeconv, get_sampled_feature, get_sampled_edgeconv_groupconv, get_edgeconv_groupconv
from tf_sampling import farthest_point_sample, gather_point


def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    k = 20
    index = farthest_point_sample(512, point_cloud)
    # index = tf.random_uniform([batch_size, 512], minval=0, maxval=num_point-1, dtype=tf.int32)
    new_point_cloud_1 = gather_point(point_cloud, index)


    net = get_sampled_edgeconv_groupconv(point_cloud, new_point_cloud_1, k, [64, 128],
                               is_training=is_training, bn_decay=bn_decay, scope='layer1', bn=True)

    k = 10
    sampled_net, new_point_cloud_2 = get_sampled_feature(new_point_cloud_1, net, 128)
    net = get_sampled_edgeconv_groupconv(net, sampled_net, k, [128, 256],
                                         is_training=is_training, bn_decay=bn_decay, scope='layer3', bn=True,
                                         sampled_pc=new_point_cloud_2, pc=new_point_cloud_1)

    net = get_edgeconv(net, k, [256], is_training=is_training, bn_decay=bn_decay, scope='layer4', bn=True,
                       associated=[sampled_net, tf.expand_dims(new_point_cloud_2, axis=-2)])

    # net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
    #                      bn=True, is_training=is_training, scope='layer5', bn_decay=bn_decay)
    # net = tf_util.conv2d(net, 512, [1, 1], padding='VALID', stride=[1, 1],
    #                      bn=True, is_training=is_training, scope='layer6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='layer7', bn_decay=bn_decay)


    net = tf.reduce_max(net, axis=1, keep_dims=True)

    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=40)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  # # np.save('./debug/input_feed.npy', input_feed)
  # input_feed = np.load('./debug/input_feed.npy')
  # print(input_feed)

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
      print(res1.shape)
      print(res1)

      print(res2.shape)
      print(res2)












