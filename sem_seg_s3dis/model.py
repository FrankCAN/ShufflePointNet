import tensorflow as tf
import math
import time
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))

import tf_util
from tf_sampling import farthest_point_sample, gather_point
from tf_interpolate import three_nn, three_interpolate

from dgcnn_util import get_sampled_edgeconv, get_edgeconv, get_sampled_feature, get_sampled_edgeconv_groupconv, pointnet_fp_module

def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32,
                   shape=(batch_size, num_point, 9))
  labels_pl = tf.placeholder(tf.int32,
                shape=(batch_size, num_point))
  return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx9 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    point_cloud_0 = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
    l0_net = tf.slice(point_cloud, [0, 0, 3], [-1, -1, 6])
    # l0_net = tf.expand_dims(l0_net, 2)

    k = 32
    npoints = 1024
    new_point_cloud_1 = gather_point(point_cloud_0, farthest_point_sample(npoints, point_cloud_0))

    net = get_sampled_edgeconv_groupconv(point_cloud_0, new_point_cloud_1, k, [32, 32, 64],
                                         is_training=is_training, bn_decay=bn_decay, scope='layer1', bn=True, is_dist=True)
    l1_net = tf.squeeze(net, [2])

    k = 20
    npoints = 256
    sampled_net, new_point_cloud_2 = get_sampled_feature(new_point_cloud_1, net, npoints)

    net = get_sampled_edgeconv_groupconv(net, sampled_net, k, [64, 64, 128],
                                         is_training=is_training, bn_decay=bn_decay, scope='layer2', bn=True,
                                         sampled_pc=new_point_cloud_2, pc=new_point_cloud_1, is_dist=True)

    net = get_edgeconv(net, k, [128], is_training=is_training, bn_decay=bn_decay, scope='layer3', bn=True,
                       associated=[sampled_net, tf.expand_dims(new_point_cloud_2, axis=-2)], is_dist=True)
    l2_net = tf.squeeze(net, [2])

    npoints = 64
    sampled_net, new_point_cloud_3 = get_sampled_feature(new_point_cloud_2, net, npoints)

    net = get_sampled_edgeconv_groupconv(net, sampled_net, k, [128, 128, 256],
                                         is_training=is_training, bn_decay=bn_decay, scope='layer4', bn=True,
                                         sampled_pc=new_point_cloud_3, pc=new_point_cloud_2, is_dist=True)

    l3_net = tf.squeeze(net, [2])

    k = 16
    npoints = 16
    sampled_net, new_point_cloud_4 = get_sampled_feature(new_point_cloud_3, net, npoints)

    net = get_sampled_edgeconv_groupconv(net, sampled_net, k, [256, 256, 512],
                                         is_training=is_training, bn_decay=bn_decay, scope='layer5', bn=True,
                                         sampled_pc=new_point_cloud_4, pc=new_point_cloud_3, is_dist=True)

    net = get_edgeconv(net, k, [512], is_training=is_training, bn_decay=bn_decay, scope='layer6', bn=True,
                       associated=[sampled_net, tf.expand_dims(new_point_cloud_4, axis=-2)], is_dist=True)
    l4_net = tf.squeeze(net, [2])

    l3_net = pointnet_fp_module(new_point_cloud_3, new_point_cloud_4, l3_net, l4_net, [256, 256], is_training, bn_decay, scope='fa_layer1', is_dist=True)
    l2_net = pointnet_fp_module(new_point_cloud_2, new_point_cloud_3, l2_net, l3_net, [256, 256], is_training, bn_decay, scope='fa_layer2', is_dist=True)
    l1_net = pointnet_fp_module(new_point_cloud_1, new_point_cloud_2, l1_net, l2_net, [256, 128], is_training, bn_decay, scope='fa_layer3', is_dist=True)
    l0_net = pointnet_fp_module(point_cloud_0, new_point_cloud_1, l0_net, l1_net, [128, 128, 128], is_training, bn_decay, scope='fa_layer4', is_dist=True)

    net = tf.expand_dims(l0_net, axis=-2)


    # FC layers
    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, is_dist=True)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv2d(net, 13, [1, 1], padding='VALID', activation_fn=None, scope='fc2', is_dist=True)
    net = tf.squeeze(net, [2])


    return net

def get_loss(pred, label):
  """ pred: B,N,13; label: B,N """
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
  return tf.reduce_mean(loss)
