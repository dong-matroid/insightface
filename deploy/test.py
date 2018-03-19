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
parser.add_argument('--image1', default = '/host/matroid/data/megaface/facescrub/myoutput/Lindsay_Hartley/Lindsay_Hartley_33155.png')
parser.add_argument('--image2', default = '/host/matroid/data/megaface/facescrub/myoutput/Lindsay_Hartley/Lindsay_Hartley_33088.png')
parser.add_argument('--image3', default = '/host/matroid/data/megaface/facescrub/myoutput/Lindsay_Hartley/Lindsay_Hartley_33088.png')
args = parser.parse_args()

model = face_embedding.FaceModel(args)
#img = cv2.imread('/raid5data/dplearn/lfw/Jude_Law/Jude_Law_0001.jpg')
img = cv2.imread(args.image1)
f1 = model.get_feature(img)
img = cv2.imread(args.image2)
f2 = model.get_feature(img)
img = cv2.imread(args.image3)
f3 = model.get_feature(img)

print(0, 1)
dist = np.linalg.norm(f1-f2) #np.sum(np.square(f1-f2))
print(dist)
sim = np.dot(f1, f2.T)
print(sim)
print(0,1,2, 'to 0,1 center')
center = (f1+f2)
center = center / np.linalg.norm(center)
dist = np.linalg.norm(f1-center) #np.sum(np.square(f1-f2))
sim = np.dot(f1, center.T)
print(dist, sim)
dist = np.linalg.norm(f2-center) #np.sum(np.square(f1-f2))
sim = np.dot(f2, center.T)
print(dist, sim)
dist = np.linalg.norm(f3-center) #np.sum(np.square(f1-f2))
sim = np.dot(f3, center.T)
print(dist, sim)


#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)

print(0, 2)
dist = np.linalg.norm(f1-f3)
print(dist)
sim = np.dot(f1, f3.T)
print(sim)

print(1, 2)
dist = np.linalg.norm(f2-f3)
print(dist)
sim = np.dot(f2, f3.T)
print(sim)
