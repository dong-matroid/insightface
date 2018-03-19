import face_embedding
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', type = str, help='path to load model.')
parser.add_argument('--keras_model', type = str, help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
keras_args = parser.parse_args(['--keras_model', 'models/model-r50-am-lfw/mxnet_resnet50.h5'])
args = parser.parse_args(['--model', 'models/model-r50-am-lfw/model,0'])

model = face_embedding.FaceModel(args)
keras_model = face_embedding.FaceModel(keras_args)
#img = cv2.imread('/raid5data/dplearn/lfw/Jude_Law/Jude_Law_0001.jpg')
img = cv2.imread('/host/matroid/data/megaface/facescrub/myoutput/Lindsay_Hartley/Lindsay_Hartley_33155.png')
f1 = model.get_feature(img)
f2 = keras_model.get_feature(img)
dist = np.sum(np.square(f1-f2))
print(dist)
