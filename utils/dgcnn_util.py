import os
import sys
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))

import tf_util
from tf_sampling import farthest_point_sample, gather_point
from tf_interpolate import three_nn, three_interpolate


def get_sampled_edgeconv(point_cloud, sampled_point_cloud, k, mlp, is_training, bn_decay, scope, bn=True, is_dist=False):
    with tf.variable_scope(scope) as sc:
        adj_matrix = tf_util.sampled_pairwise_distance(sampled_point_cloud, point_cloud)
        nn_idx = tf_util.knn(adj_matrix, k=k)

        edge_feature = tf_util.get_sampled_edge_feature(sampled_point_cloud, point_cloud, nn_idx=nn_idx, k=k)

        for i, num_out_channel in enumerate(mlp):
            edge_feature = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay, is_dist=is_dist)

        edge_feature = tf.reduce_max(edge_feature, axis=-2, keep_dims=True)

        return edge_feature

def get_sampled_edgeconv_groupconv(point_cloud, sampled_point_cloud, k, mlp, is_training, bn_decay, scope, bn=True, sampled_pc=None, pc=None, is_dist=False):
    with tf.variable_scope(scope) as sc:
        adj_matrix = tf_util.sampled_pairwise_distance(sampled_point_cloud, point_cloud)
        nn_idx = tf_util.knn(adj_matrix, k=k)

        if pc is not None:
            point_cloud = tf.concat([point_cloud, tf.expand_dims(pc, axis=-2)], axis=-1)
        if sampled_pc is not None:
            sampled_point_cloud = tf.concat([sampled_point_cloud, tf.expand_dims(sampled_pc, axis=-2)], axis=-1)


        edge_feature, neighbors, concat = tf_util.get_sampled_edge_feature_separate(sampled_point_cloud, point_cloud, nn_idx=nn_idx, k=k)

        for i, num_out_channel in enumerate(mlp):
            edge_feature = tf_util.conv2d(edge_feature, num_out_channel//2, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='edge_conv%d' % (i), bn_decay=bn_decay, is_dist=is_dist)
            neighbors = tf_util.conv2d(neighbors, num_out_channel // 2, [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=bn, is_training=is_training,
                                          scope='neig_conv%d' % (i), bn_decay=bn_decay, is_dist=is_dist)

            net = channle_shuffle(edge_feature, neighbors)

            if i < len(mlp)-1:
                ch = edge_feature.get_shape().as_list()[-1]
                edge_feature = net[:, :, :, 0:ch]
                neighbors = net[:, :, :, ch:]

        net = tf.reduce_max(net, axis=-2, keep_dims=True)

        return net


def get_edgeconv(point_cloud, k, mlp, is_training, bn_decay, scope, bn=True, associated=None, is_dist=False):
    with tf.variable_scope(scope) as sc:
        adj_matrix = tf_util.pairwise_distance(point_cloud)
        nn_idx = tf_util.knn(adj_matrix, k=k)

        if associated is not None:
            for j, feature in enumerate(associated):
                point_cloud = tf.concat([point_cloud, feature], axis=-1)

        edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

        for i, num_out_channel in enumerate(mlp):
            edge_feature = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay, is_dist=is_dist)

        edge_feature = tf.reduce_max(edge_feature, axis=-2, keep_dims=True)

        return edge_feature


def get_edgeconv_groupconv(point_cloud, k, mlp, is_training, bn_decay, scope, bn=True, associated=None):
    with tf.variable_scope(scope) as sc:
        adj_matrix = tf_util.pairwise_distance(point_cloud)
        nn_idx = tf_util.knn(adj_matrix, k=k)

        if associated is not None:
            for j, feature in enumerate(associated):
                point_cloud = tf.concat([point_cloud, feature], axis=-1)

        net = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

        for i, num_out_channel in enumerate(mlp):
            center, edge_feature = tf.split(net, num_or_size_splits=2, axis=-1)

            center = tf_util.conv2d(center, num_out_channel, [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=bn, is_training=is_training,
                                          scope='centerconv%d' % (i), bn_decay=bn_decay)

            edge_feature = tf_util.conv2d(edge_feature, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='edgeconv%d' % (i), bn_decay=bn_decay)

            net = channle_shuffle(center, edge_feature)

        edge_feature = tf.reduce_max(net, axis=-2, keep_dims=True)

        return edge_feature


def get_sampled_feature(point_cloud, net, sampled_npoints):
    batch_size = net.get_shape()[0].value
    num_feature = net.get_shape()[-1].value
    npoint = net.get_shape()[1].value

    idx_update = farthest_point_sample(sampled_npoints, point_cloud)
    # idx_update = tf.random_uniform([batch_size, sampled_npoints], minval=0, maxval=npoint-1, dtype=tf.int32)

    sampled_point_cloud = gather_point(point_cloud, idx_update)
    idx_ = tf.range(batch_size) * npoint
    idx_ = tf.reshape(idx_, [batch_size, 1])
    idx_update = idx_update + idx_
    idx_update = tf.reshape(idx_update, [-1, 1])

    new_net = tf.reshape(net, [-1, num_feature])
    new_net = tf.gather_nd(new_net, idx_update)
    new_net = tf.reshape(new_net, [batch_size, -1, 1, num_feature])

    return new_net, sampled_point_cloud


def channle_shuffle(edges, neighbors, group=2):
    """Shuffle the channel
    Args:
        inputs: 4D Tensor
        group: int, number of groups
    Returns:
        Shuffled 4D Tensor
    """
    inputs = tf.concat([neighbors, edges], axis=-1)

    in_shape = inputs.get_shape().as_list()
    h, w, in_channel = in_shape[1:]
    assert in_channel % group == 0
    l = tf.reshape(inputs, [-1, h, w, group, in_channel // group])
    l = tf.transpose(l, [0, 1, 2, 4, 3])
    l = tf.reshape(l, [-1, h, w, in_channel])

    return l


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True, is_dist=False):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay, is_dist=is_dist)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1