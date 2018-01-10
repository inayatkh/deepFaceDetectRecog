'''
Created on Nov 24, 2017

@author: inayatullah khan
@email: inayatkh@gmail.com

face verification : is this the claimed person?, it is a one to one matching problem

In general for face verification you are given two images and the objectives is to tell whether they are of 
of the same person or not, 
'''
# switch off warnings

import sys
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer

from utils import faceNetModel

from utils import load_weights_from_FaceNet

from utils import loss_triplet

from utils import img_to_encoding

from keras import backend as KB
KB.set_image_data_format('channels_first')
# We can also set the data_format="channels_first" by passing it  as an argument to the conv and maxpool layer

from utils import load_weights

from utils import mtcnn

#from utils import normalize_faces_landmarks
from utils import align_face
from utils import build_montages


#

import cv2

import numpy as np

import h5py

VERBOSE = True
#import datetime

def debug(msg, msgType="[INFO]"):
    '''
      the _debug  method, which can be used to (optionally) write debugging messages
    '''
    # check to see if the message should be printed
    if VERBOSE:
        print("**** {} {} -".format('\033[92m' + msgType + '\033[0m', msg))
        
    
def compute_dist_label(reg_faces_encodings, index_dic, test_face_encodings, THRESHOLD=0.7 ):
    
    
    # Calculate Euclidean distances between face encodings calculated on face dectected
    # in test image with all the faces encodings calculated while registering persons
    
    print("test_face_encodings shape", test_face_encodings.shape)
    print("reg_faces_encodings shape", reg_faces_encodings.shape)
    distances = np.linalg.norm(test_face_encodings -reg_faces_encodings , axis=1, keepdims=True)
    print("distances shape", distances.shape)
    # Calculate minimum distance and index of this face
    argmin = np.argmin(distances)  # index
    
    minDistance = distances[argmin]  # minimum distance
    
    # In general, if two face encodings vectors have a Euclidean
    # distance between them less than 0.6 then they are from the same
    # person, otherwise they are from different people.
    
    # This threshold will vary depending upon number of images enrolled
    # and various variations (illuminaton, camera quality) between
    # enrolled images and query image
    # We are using a threshold of 0.5
    
    # If minimum distance if less than threshold
    # find the name of person from index
    # else the person in query image is unknown
    
    
    debug("minDistance = {} , armin={}, registered face ={}".format(minDistance, argmin, index_dic[argmin]))
    
    if minDistance <= THRESHOLD:
        #label = os.path.splitext(os.path.dirname(index_dic[argmin]).split('/')[-1])[0]
        label = index_dic[argmin]
    else:
        label = 'unknown'
        
    
    return (minDistance, label, argmin)
if __name__ == '__main__':
    
    np.set_printoptions(threshold=np.nan)
    
    debug("Initializing Deep faceNet recognition model ....")
    faceNet = faceNetModel(input_shape=(3,96,96))
    
    debug(" Total Params of the faceNet model: {}".format(faceNet.count_params()))
    
    '''
     FaceNet is trained by minimizing the triplet loss. However, since training
     requires a large amount of images and heavy computation. 
     Therefore, we load a previously trained model
     
    '''
    debug("Loading Weights of FaceNet model")
    #faceNet.compile(optimizer='adam', loss=loss_triplet(alpha=0.2), metrics=['accuracy'])
    faceNet.compile(optimizer='adam', loss=loss_triplet, metrics=['accuracy'])
    faceNet=load_weights_from_FaceNet(faceNet)
    debug(" weights are loaded from the csv files")
    
    '''
    MTCNN face detection
    
    '''
    
    
    
    #threshold = [0.6,0.6,0.7]
    
    debug("Loading Deep face Detector (MTCNN) recognition model ....")
    threshold = [0.3,0.7,0.7]
    faceDetector = mtcnn()
    
    
    
    debug("Loading Registered Persons faces encodings and index files generated with registerPerson.py ....")
    
    REG_PERSONS_PATH = './images/registeredfaces'
    
    HDF5_FACE_DATASET_NAME = "registeredFaces.h5"
    
    hdf5_dataset_fullName = os.path.join(REG_PERSONS_PATH, HDF5_FACE_DATASET_NAME)
    
    hdf5_file = h5py.File(hdf5_dataset_fullName, 'r')
    
    index_dic = hdf5_file["index_dic"][:]
    reg_faces_encodings = hdf5_file["faces_encoding"][:]
    
    reg_aligned_faces = hdf5_file["face_imgs_cv"]
    
    
    #hdf5_file.close()
    
    #print(len(reg_aligned_faces), type(reg_aligned_faces))
    
    #sys.exit()

    
    
    test_images_path = './images/test'
    
    test_images_result_path = './images/result-test'
    
   
    
    
    if not os.path.isdir(test_images_result_path):
        os.mkdir(test_images_result_path)
    
    dispWin = "Deep Face Detection (MTCNN) and Recognition (FaceNet + Euc dist)"
    #cv2.namedWindow(dispWin, cv2.WINDOW_NORMAL)
    #cv2.namedWindow(dispWin, flags= cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO )
    cv2.namedWindow(dispWin, flags= cv2.WINDOW_AUTOSIZE | cv2.WINDOW_FREERATIO )
    
    
    
    for img_file_name in os.listdir(test_images_path):
        img_file_path = os.path.join(test_images_path, img_file_name)
        if not os.path.isdir(img_file_path):
            test_img = cv2.imread(img_file_path)
            test_img_draw = test_img.copy()
            thickness = (test_img.shape[0] + test_img.shape[1]) // 350
            debug('detecting face in file : {}'.format(img_file_path))
            
            bboxes_lndmarks = faceDetector.detectFace(test_img, threshold)
            
            aligned_faces_list=[]
            test_face_img_list =[]
            
            matched_face_img_list =[]
            
            for bbox_lndmarks in bboxes_lndmarks:

                if bbox_lndmarks is not None:

                    left, top,  right, bottom = bbox_lndmarks[0], bbox_lndmarks[1], bbox_lndmarks[2], bbox_lndmarks[3]
                    
                    left =int(left)
                    top = int(top)
                    right = int(right)
                    bottom = int(bottom)
                    
                    '''
                    face_bbox_margin = (right - left)/4

                    #top = max(0, np.floor(top + face_bbox_margin).astype('int32'))
                    #left = max(0, np.floor(left + face_bbox_margin).astype('int32'))
                    top = max(0, np.floor(top - face_bbox_margin).astype('int32'))
                    left = max(0, np.floor(left - face_bbox_margin).astype('int32'))
                    bottom = min(test_img.shape[0], np.floor(bottom + face_bbox_margin).astype('int32'))
                    right = min(test_img.shape[1], np.floor(right + face_bbox_margin).astype('int32'))
                    
                    '''

                    #face_bboxes.append([top, left, bottom, right])
                    test_face_img = test_img[top:bottom, left:right]
                    test_face_img_list.append(test_face_img)
                    
                    test_face_lndmarks =[]
                    for i in range(5, 15, 2):
                        #test_face_lndmarks.append([int(bbox_lndmarks[i + 0] - left), int(bbox_lndmarks[i + 1]-top)])
                        test_face_lndmarks.append([int(bbox_lndmarks[i + 0]), int(bbox_lndmarks[i + 1])])
                        
                    
                    test_face_lndmarks = np.array(test_face_lndmarks)
                    
                    ### encoding test_face
                    
                    #test_face_img = cv2.resize(test_face_img, (96,96), interpolation=cv2.INTER_CUBIC)
                    #normalized_face_img, normalized_face_lndmarks = normalize_faces_landmarks(out_img_size=(96,96),
                    #                                                                          in_img=test_face_img,
                    #                                                                          landmarks=test_face_lndmarks)
                    
                    normalized_face_img, normalized_face_lndmarks = align_face(out_img_size=(96,96),
                                                                                              in_img=test_img,
                                                                                              landmarks=test_face_lndmarks)
                    
                    aligned_faces_list.append(normalized_face_img)
                    
                    #test_face_encodings = img_to_encoding(test_face_img, faceNet)
                    test_face_encodings = img_to_encoding(normalized_face_img, faceNet)
                    dist, identity, matched_idx=compute_dist_label(reg_faces_encodings, index_dic, test_face_encodings, THRESHOLD=0.76)
                    
                    
                    
                    #cv2.imwrite(identity+".jpg", test_face_img)
                    
                    ######## drawing faces and labels or person names
                    
                    if(identity == "unknown"):
                        DRAW_COLOR = (255, 255, 0)
                        unimg=np.zeros((96,96,3), dtype= np.uint)
                        matched_face_img_list.append(unimg)
                        
                    else:
                        matched_face_img_list.append(reg_aligned_faces[matched_idx])
                        DRAW_COLOR = (0, 255, 0)
                        
                    start_pt = (int(bbox_lndmarks[0]), int(bbox_lndmarks[1]))
                    end_pt = (int(bbox_lndmarks[2]), int(bbox_lndmarks[3]))
                    for i in range(thickness):
                        start_pt = (int(bbox_lndmarks[0])+i, int(bbox_lndmarks[1])+i)
                        end_pt = (int(bbox_lndmarks[2])-i, int(bbox_lndmarks[3])-i)
                        cv2.rectangle(test_img_draw, start_pt, end_pt, DRAW_COLOR, 1)
                        
                    for i in range(5, 15, 2):
                        cv2.circle(test_img_draw, (int(bbox_lndmarks[i + 0]), int(bbox_lndmarks[i + 1])), 2, (0, 255, 0))
                    
                    
                    person_id_tag = "%s: %.2f" % (identity, dist)
                                        
                    text_end_pt   = (int(bbox_lndmarks[0]) +  len(person_id_tag) * 10, int(bbox_lndmarks[1]) - 20 )
                    text_start_pt = (max(int(bbox_lndmarks[0]), 10), max(int(bbox_lndmarks[1]), 10))
                    
                    cv2.rectangle(test_img_draw,text_start_pt , text_end_pt,
                                  DRAW_COLOR, -1, cv2.LINE_AA)
                    cv2.putText(test_img_draw, person_id_tag, text_start_pt, cv2.FONT_ITALIC,0.4, (255, 255, 255), 1)
            
            
        
            cv2.imshow(dispWin, test_img_draw)
            rest_img_file_path = os.path.join(test_images_result_path, img_file_name)
            cv2.imwrite(rest_img_file_path, test_img_draw)
            
            if len(aligned_faces_list) > 0 :
                
                montage_shape = (len(aligned_faces_list),1)
                
                faces_montages = build_montages(aligned_faces_list, image_shape=(96,96),
                                                montage_shape=montage_shape)
                for num, faces_montage in enumerate(faces_montages):
                    cv2.imshow("Aligned Faces" + "  " + str(num), faces_montage)
                    
                montage_shape = (len(test_face_img_list),1)
                
                test_faces_montages = build_montages(test_face_img_list, image_shape=(96,96),
                                                     montage_shape=montage_shape)
                for num, test_faces_montage in enumerate(test_faces_montages):
                    cv2.imshow("Original Faces" + "  " + str(num), test_faces_montage)
                    
                reg_matched_faces_montages = build_montages(matched_face_img_list, image_shape=(96,96),
                                                            montage_shape=montage_shape)
                
                for num, reg_matched_faces_montage in enumerate(reg_matched_faces_montages):
                    cv2.imshow("Matched Registered Faces" + "  " + str(num), reg_matched_faces_montage)
                
            
            
                
            if cv2.waitKey(0) == ord('q'):
                break
            
                
            
    
    cv2.destroyAllWindows()
    hdf5_file.close()
    ######################################
    
    
                
    
    
