# Copyright 2017 Chenxi Liu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# sample usage:
# python deeplab_main.py 0 single

import tensorflow as tf
import numpy as np
from deeplab_model import DeepLab
from PIL import Image
import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import pdb
import time

def process_im(imname, mu):
  im = np.array(Image.open(imname), dtype=np.float32)
  if im.ndim == 3:
    if im.shape[2] == 4:
      im = im[:, :, 0:3]
    im = im[:,:,::-1]
  else:
    im = np.tile(im[:, :, np.newaxis], (1, 1, 3))
  im -= mu
  im = np.expand_dims(im, axis=0)
  return im

if __name__ == "__main__":

  mu = np.array((104.00698793, 116.66876762, 122.67891434))

  if sys.argv[2] == 'train':
    pretrained_model = './model/ResNet101_init.tfmodel'
    model = DeepLab(mode='train',weight_decay_rate=1e-7)
    load_var = {var.op.name: var for var in tf.global_variables() 
        if not 'Momentum' in var.op.name and not 'global_step' in var.op.name}
    snapshot_restorer = tf.train.Saver(load_var)
  else:
    # pretrained_model = './model/ResNet101_train.tfmodel'
    pretrained_model = './model/ResNet101_epoch_2.tfmodel'
    model = DeepLab()
    snapshot_restorer = tf.train.Saver()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  snapshot_restorer.restore(sess, pretrained_model)

  if sys.argv[2] == 'single':
    im = process_im('example/2007_000129.jpg', mu)
    pred = sess.run(model.up, feed_dict={
              model.images  : im
          })
    pred = np.argmax(pred, axis=3).squeeze().astype(np.uint8)
    seg = Image.fromarray(pred)
    seg.save('example/2007_000129.png')

  elif sys.argv[2] == 'test':
    pascal_dir = '/home/VOCdevkit/'
    lines = np.loadtxt(pascal_dir + 'test.txt', dtype=str)
    for i, line in enumerate(lines):
      imname = line
      im = process_im(pascal_dir + "JPEGImages/" + imname + '.jpg', mu)
      pred = sess.run(model.up, feed_dict={
                model.images : im
            })
      pred = np.argmax(pred, axis=3).squeeze().astype(np.uint8)
      seg = Image.fromarray(pred)
      seg.save('example/test/' + imname + '.png')
      print('processing %d/%d' % (i + 1, len(lines)))
      sys.stdout.flush()

  elif sys.argv[2] == 'train':
    cls_loss_avg = 0
    decay = 0.99
    num_epochs = 2 # train for 2 epochs
    snapshot_saver = tf.train.Saver(max_to_keep = 1000)
    snapshot_file = './model/ResNet101_epoch_%d.tfmodel'
    pascal_dir = '/home/VOCdevkit'
    lines = np.loadtxt(pascal_dir + '/train.txt', dtype=str)
    for epoch in range(num_epochs):
      lines = np.random.permutation(lines)
      for i, line in enumerate(lines):
        btime = time.time()
        imname, labelname = line
        im = process_im(pascal_dir + imname, mu)
        label = np.array(Image.open(pascal_dir + labelname))
        label = np.expand_dims(label, axis=0)
        _, cls_loss_val, lr_val, label_val = sess.run([model.train_step,
          model.cls_loss,
          model.learning_rate,
          model.labels_coarse],
          feed_dict={
            model.images : im,
            model.labels : np.expand_dims(label, axis=3)
          })
        cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val
        print('runtime = %fs, iter = %d / %d, loss (cur) = %f, loss (avg) = %f, lr = %f' % (time.time()-btime, i, len(lines), cls_loss_val, cls_loss_avg, lr_val))
        sys.stdout.flush()
      snapshot_saver.save(sess, snapshot_file % (epoch + 1))
