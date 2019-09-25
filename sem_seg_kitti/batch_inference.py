import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from model import *
import indoor3d_util
import provider
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=16384, help='Point number [default: 4096]')
parser.add_argument('--model_path', default='log1/epoch_80.ckpt', help='model checkpoint file path')
parser.add_argument('--dump_dir', default='log1/dump', help='dump folder path')
parser.add_argument('--output_filelist', default='log1/output_filelist.txt', help='TXT filename, filelist, each line is an output for a room')
parser.add_argument('--room_data_filelist', required=False, help='TXT filename, filelist, each line is a test room data label file.')
parser.add_argument('--no_clutter', action='store_true', help='If true, donot count the clutter class')
parser.add_argument('--visu', action='store_true', help='Whether to output OBJ file for prediction visualization.')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

NUM_CLASSES = 4

LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


test_data, test_label = provider.load_h5('/home/chencan/dataset/kitti/test_data.h5')
test_label = test_label.reshape((-1, NUM_POINT))



def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():

    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        pred = get_model(pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)

        saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}

    total_correct = 0
    total_seen = 0

    a, b = eval_one_epoch(sess, ops)
    total_correct += a
    total_seen += b

    log_string('all room eval accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    current_data = test_data
    current_label = test_label

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    print(file_size)
    print('num_batches:  ', num_batches)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx

        out_data_label_filename = str(batch_idx) + '_pred.txt'
        out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
        out_gt_label_filename = str(batch_idx) + '_gt.txt'
        out_gt_label_filename = os.path.join(DUMP_DIR, out_gt_label_filename)

        fout_data_label = open(out_data_label_filename, 'w')
        fout_gt_label = open(out_gt_label_filename, 'w')


        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']],
                                      feed_dict=feed_dict)

        if FLAGS.no_clutter:
            pred_label = np.argmax(pred_val[:, :, 0:12], 2)  # BxN
        else:
            pred_label = np.argmax(pred_val, 2)  # BxN

        max_room_x = max(current_data[start_idx:end_idx, :, 0])
        max_room_y = max(current_data[start_idx:end_idx, :, 1])
        max_room_z = max(current_data[start_idx:end_idx, :, 2])

        for b in range(BATCH_SIZE):
            pts = current_data[start_idx + b, :, :]
            l = current_label[start_idx + b, :]
            pts[:, 0] *= max_room_x
            pts[:, 1] *= max_room_y
            pts[:, 2] *= max_room_z

            pred = pred_label[b, :]
            for i in range(NUM_POINT):
                fout_data_label.write('%d\n' % (pred[i]))
                fout_gt_label.write('%d\n' % (l[i]))

        correct = np.sum(pred_label == current_label[start_idx:end_idx, :])
        total_correct += correct
        total_seen += (cur_batch_size * NUM_POINT)
        loss_sum += (loss_val * BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[i - start_idx, j] == l)

        fout_data_label.close()
        fout_gt_label.close()

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen / NUM_POINT)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))

    return total_correct, total_seen


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
