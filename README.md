# Go Wider: An Efficient Neural Network for Point Cloud Analysis via Group Convolutions
created by Can Chen, Luca Zanotti Fragonara, Antonios Tsourdos from Cranfield University

[[Paper]](https://arxiv.org/abs/1909.10431)

# Overview
We propose a deep-wide neural network, called ShufflePointNet, to exploit fine-grained local features and reduce redundancy in parallel using group convolution and channel shuffle operation. Experiments show state-of-the-art performance in both shape classification and semantic segmentation tasks.

In this repository, we release code for training a ShufflePointNet classification network on ModelNet40 dataset and segmentation network on ShapeNet part, S3DIS, KITTI dataset.

# Installation
The code is tested under [TensorFlow](https://www.tensorflow.org/) 1.6 GPU version and Python 3.5 on Ubuntu 16.04. some dependencies for a few Python libraries should also be installed in advance, such as cv2, h5py.

Some necessary operators are included under tf_ops, so you need to compile them in advance. Update nvcc and python path if necessary. The code is tested under TF1.6.0. If you are using earlier version it's possible that you need to remove the -D_GLIBCXX_USE_CXX11_ABI=0 flag in g++ command in order to compile correctly.

