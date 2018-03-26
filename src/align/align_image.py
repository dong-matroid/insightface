"""align a given image, output the face crop and the points"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import json
import argparse
import tensorflow as tf
import numpy as np
#import facenet
import detect_face
import random
from time import sleep
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_image
from skimage import transform as trans
import cv2

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def IOU(Reframe,GTframe):
  x1 = Reframe[0];
  y1 = Reframe[1];
  width1 = Reframe[2]-Reframe[0];
  height1 = Reframe[3]-Reframe[1];

  x2 = GTframe[0]
  y2 = GTframe[1]
  width2 = GTframe[2]-GTframe[0]
  height2 = GTframe[3]-GTframe[1]

  endx = max(x1+width1,x2+width2)
  startx = min(x1,x2)
  width = width1+width2-(endx-startx)

  endy = max(y1+height1,y2+height2)
  starty = min(y1,y2)
  height = height1+height2-(endy-starty)

  if width <=0 or height <= 0:
    ratio = 0
  else:
    Area = width*height
    Area1 = width1*height1
    Area2 = width2*height2
    ratio = Area*1./(Area1+Area2-Area)
  return ratio


def main(args):
    output_dir = os.path.expanduser(args.output_dir)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    minsize = 100 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    image_size = [112,96]
    image_size = [112,112]
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    if image_size[1]==112:
      src[:,0] += 8.0

    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)

    image_path = args.input_image
    if not os.path.exists(image_path):
      print('image not found (%s)'%image_path)
      return

    try:
        img = misc.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)
    else:
        if img.ndim<2:
            print('Unable to align "%s", img dim error' % image_path)
            #text_file.write('%s\n' % (output_filename))
            return
        if img.ndim == 2:
            img = to_rgb(img)
        img = img[:,:,0:3]
        target_file = os.path.join(args.output_dir, os.path.basename(image_path))

        _minsize = 40
        bounding_boxes, points = detect_face.detect_face(img, 40, pnet, rnet, onet, threshold, factor)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for bindex in range(bounding_boxes.shape[0]):
          det = bounding_boxes[bindex,0:4]
          dst = points[:, bindex].reshape( (2,5) ).T
          print(dst)
          for p in dst:
            cv2.circle(bgr_img, tuple(p), 1, (0,0,255), -1)

          tform = trans.SimilarityTransform()
          tform.estimate(dst, src)
          M = tform.params[0:2,:]
          warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
          bgr = warped[...,::-1]
          cv2.imwrite(target_file + '-%s.png' %  + bindex, bgr)
        cv2.imwrite(target_file + '-points.png', bgr_img)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-image', type=str, help='Directory with unaligned images.')
    parser.add_argument('--output-dir', type=str, help='Directory with aligned face thumbnails.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
