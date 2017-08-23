#!/bin/env python
#-*- encoding:UTF-8 -*-

import BaseHTTPServer
from SocketServer import ThreadingMixIn
import json
import traceback
import shutil
import re
import os
import sys
import time

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from PIL import Image

import tensorflow as tf
import numpy as np
import base64
import argparse

import tensorflow as tf
import numpy as np
from deeplab_model import DeepLab
from deeplab_main import process_im_ex, save_image_ex

import httpclient

rePath = re.compile(r'[^\w]+')

args = None
model = None
sess = None
srvr = None
step = 0
savedir = './snapshots'
stepfile = os.path.join(savedir, 'model.ckpt.step')
mu = np.array((104.00698793, 116.66876762, 122.67891434))

class HTTPRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def version_string(self):
        return 'Tensorflow-DeepLab-ResNet/0.1'

    def send_except(self):
        f = StringIO()
        traceback.print_exc(file=f)
        length = f.tell()
        f.seek(0)
        self.send_response(500)
        self.send_header('Content-Type', "text/plain; charset=utf-8")
        self.send_header('Content-Length', length)
        self.end_headers()
        shutil.copyfileobj(f, self.wfile)
        f.close()

    def do_POST(self):
        contentType = self.headers.getheader('content-type')
        if contentType is None:
            self.send_error(420, 'The content-type request header was not found')
            return
        mimeType, _ = contentType.split(';')
        if mimeType != 'application/json':
            self.send_error(415, 'Unsupported content-type')
            return

        contentLength = self.headers.getheader('content-length')
        if contentLength is None:
            self.send_error(421, 'The content-length request header was not found')
            return
        length = int(contentLength)
        if length <= 0:
            self.send_error(422, 'Content-Length request headers must be greater than 0')
            return
        postStr = self.rfile.read(length);
        try:
            post = json.loads(postStr)
        except:
            self.send_except()
            return

        action = 'action_' + rePath.sub('_', self.path).strip('_')
        if hasattr(self, action):
            method = getattr(self, action)
            try:
                self.post = post
                post = method(**post)
            except:
                self.send_except()
                return

        postStr = json.dumps(post, ensure_ascii=False, indent=4, separators=(',', ': ')).encode('utf-8')

        self.send_response(200)
        self.send_header('Content-Type', "application/json; charset=utf-8")
        self.send_header('Content-Length', len(postStr))
        self.end_headers()

        self.wfile.write(postStr)

    def action_train(self, image, label):
        global model
        global sess
        global mu

        image = Image.open(StringIO(base64.decodestring(image)))
        image = process_im_ex(image, mu)
        label = Image.open(StringIO(base64.decodestring(label)))
        label = np.array(label)
        label = np.expand_dims(label, axis=0)
        label = np.expand_dims(label, axis=3)

        cls_loss_avg = 0
        decay = 0.99
        trains = 10
        f = StringIO()
        for i in range(trains):
            btime = time.time()
            _, cls_loss_val, lr_val, label_val = sess.run(
                [
                    model.train_step,
                    model.cls_loss,
                    model.learning_rate,
                    model.labels_coarse
                ],
                feed_dict={
                    model.images : image,
                    model.labels : label
                }
            )
            cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val
            msg = 'runtime = %2.3fs, iter = %d / %d, loss (cur) = %f, loss (avg) = %f, lr = %f\n' % (time.time()-btime, i, trains, cls_loss_val, cls_loss_avg, lr_val)
            f.write(msg)
            sys.stdout.write(msg)
            sys.stdout.flush()
        length = f.tell()
        f.seek(0)
        ret = base64.encodestring(f.read(length))
        f.close()
        return ret

    def action_test(self, image):
        global model
        global sess
        global mu

        image = Image.open(StringIO(base64.decodestring(image)))
        image = process_im_ex(image, mu)
        pred = sess.run(model.up, feed_dict={
            model.images : image
        })
        pred = np.argmax(pred, axis=3).squeeze().astype(np.uint8)
        f = StringIO()
        save_image_ex(pred).save(f, format='PNG')
        length = f.tell()
        f.seek(0)
        ret = base64.encodestring(f.read(length))
        f.close()
        return ret

class ThreadingServer(ThreadingMixIn, BaseHTTPServer.HTTPServer):
    pass

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Deeplab Resnet Http Server.")
    parser.add_argument("--model-weights", type=str, default=None, help="Path to the file with model weights.")
    parser.add_argument("--port", type=int, default=8000, help="Listen on port(default: 8000)")
    return parser.parse_args()

def main():
    """Create the model and start the evaluation process."""
    global args
    global model
    global sess
    global srvr
    global savedir
    global stepfile
    global step

    args = get_arguments()
    model = DeepLab(mode='train',weight_decay_rate=1e-7)
    load_var = {var.op.name: var for var in tf.global_variables() 
        if not 'Momentum' in var.op.name and not 'global_step' in var.op.name}

    # Set up TF session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    sess.run(tf.global_variables_initializer())
    
    if os.path.exists(stepfile):
        step = int(httpclient.readfile(stepfile))
    if args.model_weights is None:
        args.model_weights = './model/ResNet101_train.tfmodel'
        indexfile = os.path.join(savedir, 'model.ckpt-%d.index' % step)
        if os.path.exists(indexfile):
            args.model_weights = indexfile[:-6]

    # Load weights.
    loader = tf.train.Saver(load_var)
    print 'Restoring from "%s.*" ...' % args.model_weights
    loader.restore(sess, args.model_weights)

    #单线程
    # srvr = BaseHTTPServer.HTTPServer(("0.0.0.0", args.port), SimpleHTTPRequestHandler)
    
    #多线程
    srvr = ThreadingServer(("0.0.0.0", args.port), HTTPRequestHandler)

    print "serving at port", args.port
    srvr.serve_forever()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # save train result
        if sess is not None:
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            step += 1
            print 'Saving to "%s" ...' % os.path.join(savedir, 'model.ckpt-%d.*' % step)
            httpclient.writefile(stepfile, str(step))
            loader = tf.train.Saver(max_to_keep = 1000)
            loader.save(sess, os.path.join(savedir, 'model.ckpt'), global_step=step)
        if srvr is not None:
            srvr.shutdown()
        print 'exited.'
