import tensorflow as tf
import numpy as np
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))

from dgcnn_util import get_sampled_edgeconv, get_edgeconv, get_sampled_feature, get_sampled_edgeconv_groupconv, pointnet_fp_module

import tf_util
from tf_sampling import farthest_point_sample, gather_point
from tf_interpolate import three_nn, three_interpolate

NUM_CATEGORIES = 16

def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
    batch_size, num_point, weight_decay, bn_decay=None):
    point_cloud_0 = point_cloud
    l0_net = point_cloud
    # l0_net = tf.expand_dims(l0_net, 2)

    k = 32
    npoints = 1024
    new_point_cloud_1 = gather_point(point_cloud_0, farthest_point_sample(npoints, point_cloud_0))

    net = get_sampled_edgeconv_groupconv(point_cloud_0, new_point_cloud_1, k, [32, 32, 64],
                                         is_training=is_training, bn_decay=bn_decay, scope='layer1', bn=True)
    l1_net = tf.squeeze(net, [2])

    k = 20
    npoints = 256
    sampled_net, new_point_cloud_2 = get_sampled_feature(new_point_cloud_1, net, npoints)

    net = get_sampled_edgeconv_groupconv(net, sampled_net, k, [64, 64, 128],
                                         is_training=is_training, bn_decay=bn_decay, scope='layer2', bn=True,
                                         sampled_pc=new_point_cloud_2, pc=new_point_cloud_1)

    net = get_edgeconv(net, k, [128], is_training=is_training, bn_decay=bn_decay, scope='layer3', bn=True,
                       associated=[sampled_net, tf.expand_dims(new_point_cloud_2, axis=-2)])
    l2_net = tf.squeeze(net, [2])

    npoints = 64
    sampled_net, new_point_cloud_3 = get_sampled_feature(new_point_cloud_2, net, npoints)

    net = get_sampled_edgeconv_groupconv(net, sampled_net, k, [128, 128, 256],
                                         is_training=is_training, bn_decay=bn_decay, scope='layer4', bn=True,
                                         sampled_pc=new_point_cloud_3, pc=new_point_cloud_2)

    l3_net = tf.squeeze(net, [2])

    k = 16
    npoints = 16
    sampled_net, new_point_cloud_4 = get_sampled_feature(new_point_cloud_3, net, npoints)

    net = get_sampled_edgeconv_groupconv(net, sampled_net, k, [256, 256, 512],
                                         is_training=is_training, bn_decay=bn_decay, scope='layer5', bn=True,
                                         sampled_pc=new_point_cloud_4, pc=new_point_cloud_3)

    net = get_edgeconv(net, k, [512], is_training=is_training, bn_decay=bn_decay, scope='layer6', bn=True,
                       associated=[sampled_net, tf.expand_dims(new_point_cloud_4, axis=-2)])
    l4_net = tf.squeeze(net, [2])

    l3_net = pointnet_fp_module(new_point_cloud_3, new_point_cloud_4, l3_net, l4_net, [256, 256], is_training, bn_decay,
                                scope='fa_layer1')
    l2_net = pointnet_fp_module(new_point_cloud_2, new_point_cloud_3, l2_net, l3_net, [256, 256], is_training, bn_decay,
                                scope='fa_layer2')
    l1_net = pointnet_fp_module(new_point_cloud_1, new_point_cloud_2, l1_net, l2_net, [256, 128], is_training, bn_decay,
                                scope='fa_layer3')
    l0_net = pointnet_fp_module(point_cloud_0, new_point_cloud_1, l0_net, l1_net, [128, 128, 128], is_training,
                                bn_decay, scope='fa_layer4')

    net = tf.expand_dims(l0_net, axis=-2)

    out_max = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')

    one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
    one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=True, is_training=is_training,
                                          scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
    out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
    expand = tf.tile(out_max, [1, num_point, 1, 1])

    concat = tf.concat(axis=3, values=[expand, net])

    net2 = tf_util.conv2d(concat, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
    net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
    net2 = tf_util.conv2d(net2, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
    net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
    net2 = tf_util.conv2d(net2, 128, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
    net2 = tf_util.conv2d(net2, part_num, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None,
                          bn=False, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)

    net2 = tf.reshape(net2, [batch_size, num_point, part_num])

    return net2







def get_loss(seg_pred, seg):
    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg),
                                           axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)
    per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res
