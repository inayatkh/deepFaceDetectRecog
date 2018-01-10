'''
Created on Dec 5, 2017

@author: Inayat Khan
@email: inayatkh@gmail.com
'''


import os 
import imutils
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'




#On python3.x cPickle has changed from cPickle to _pickle
import _pickle as cPickle


from keras.models import Model

from utils import faceNetModel


from keras import backend as KB
KB.set_image_data_format('channels_first')
# We can also set the data_format="channels_first" by passing it  as an argument to the conv and maxpool layer

from utils import load_weights_from_FaceNet
#from utils import normalize_faces_landmarks
from utils import align_face

from utils import loss_triplet

from utils import img_to_encoding

from utils import extract_draw_faces

from utils import mtcnn


import h5py

#

import cv2

import numpy as np



VERBOSE = True
#import datetime

FACES_IMAGES_PATH = "./images/faces/"
OUT_REG_FACES_PATH = "./images/registeredfaces"

HDF5_FACE_DATASET_NAME = "registeredFaces.h5"


def save_encoded_faces_pickle(images_path,out_pic_file_path, faceNet):
    '''
      load face images from the images_path, detect faces, encode them and save the results pickle file

    '''
    threshold = [0.3,0.6,0.7]
    faceDetector_mtcnn = mtcnn()

    # prepare our registered people data
    # assuming images folder contain individual images in subfolders

    # each person images are put in corresponding subfolders, and each image of a person contains only face

    person_subfolders = []

    for p in os.listdir(images_path):
        ppath = os.path.join(images_path, p)
        if os.path.isdir(ppath):
            person_subfolders.append(ppath)

    #initial person name label Map
    name_label_map = {}
    # integer values of each person label
    labels =[]

    person_image_paths=[]

    for i, person_subfolder in enumerate(person_subfolders):
        for p in os.listdir(person_subfolder):
            ppath = os.path.join(person_subfolder, p)
            if p.endswith('jpg'):
                person_image_paths.append(ppath)
                labels.append(i)
                name_label_map[p] = person_subfolder

    
    # now process each person images one by one and encode them using the faceNet model
    # we will stored each encodings in face_encodings and their corresponding labels in index dictionary

    index_dic = {}
    i =0
    faces_encoding = None

    for image_path in person_image_paths:
        debug("encoding face detected in : {}".format(image_path))

        image_cv = cv2.imread(image_path, 1)

        img_draw, face_imgs, face_bboxes, faces_landmarks = extract_draw_faces(faceDetector_mtcnn, image_cv, threshold)
        
        cv2.imshow("face image", img_draw)
        
        
        
        
        if len(face_imgs) == 0:
            continue
        
        face_lndmrk_draw = face_imgs[0].copy()
        
        lndmrks = np.array(faces_landmarks[0])
        
        cv2.putText(face_lndmrk_draw, '1', (lndmrks[0][0],lndmrks[0][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(face_lndmrk_draw, (lndmrks[0][0],lndmrks[0][1]), 2, (0, 255, 0))
        
        cv2.putText(face_lndmrk_draw, '2', (lndmrks[1][0], lndmrks[1][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(face_lndmrk_draw, (lndmrks[1][0],lndmrks[1][1]), 2, (0, 255, 0))
        
        cv2.putText(face_lndmrk_draw, '3', (lndmrks[2][0], lndmrks[2][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(face_lndmrk_draw, (lndmrks[2][0],lndmrks[2][1]), 2, (0, 255, 0))
        
        cv2.putText(face_lndmrk_draw, '4', (lndmrks[3][0], lndmrks[3][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(face_lndmrk_draw, (lndmrks[3][0],lndmrks[3][1]), 2, (0, 255, 0))
        
        cv2.putText(face_lndmrk_draw, '5', (lndmrks[4][0], lndmrks[4][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(face_lndmrk_draw, (lndmrks[4][0],lndmrks[4][1]), 2, (0, 255, 0))
        
        cv2.imshow("face", face_lndmrk_draw)
        
        
        
        #face_cv =  cv2.resize(face_imgs[0], (96,96), interpolation=cv2.INTER_CUBIC)
        
        #cv2.imshow("face 96x96", face_cv)
        #aligned_face_cv, aligned_face_lndmarks = normalize_faces_landmarks(out_img_size=(96,96), in_img=face_imgs[0], landmarks=lndmrks)
        #aligned_face_cv, aligned_face_lndmarks = align_face(out_img_size=(96,96), in_img=face_imgs[0], landmarks=lndmrks)
        
        aligned_face_cv, aligned_face_lndmarks = align_face(out_img_size=(96,96), in_img=image_cv, landmarks=lndmrks)
        
        aligned_face_draw = aligned_face_cv.copy()

        
        cv2.putText(aligned_face_draw, '1', (aligned_face_lndmarks[0][0],aligned_face_lndmarks[0][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(aligned_face_draw,  (aligned_face_lndmarks[0][0],aligned_face_lndmarks[0][1]), 2, (0, 255, 0))
        cv2.putText(aligned_face_draw, '2', (aligned_face_lndmarks[1][0], aligned_face_lndmarks[1][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(aligned_face_draw,  (aligned_face_lndmarks[1][0],aligned_face_lndmarks[1][1]), 2, (0, 255, 0))
        
        cv2.imshow("aligned", aligned_face_draw)
        
        cv2.waitKey(0)
        
        # we have assumed that a single image will have only the face of the person
        
        # compute the encodings
        
        
        #f_encoding = img_to_encoding(face_cv, faceNet)
        
        f_encoding = img_to_encoding(aligned_face_cv,faceNet)
        
        if faces_encoding is None:
            faces_encoding = f_encoding
        else:
            faces_encoding = np.concatenate((faces_encoding, f_encoding), axis=0)
            
        # save the label for this face in index_dic
        # later, this will be used for verification or identification of a person
        index_dic[i] = image_path
        i = i+1
        
        
        
    # save the face encodings and label index dict
    
    np.save(os.path.join(out_pic_file_path,'faces_encoding.npy'), faces_encoding)
    
    with open(os.path.join(out_pic_file_path,'index_dic.pkl'),'wb') as f :
        cPickle.dump(index_dic, f)
        
        
            
        
        
        
            
        
    
    
def debug(msg, msgType="[INFO]"):
    '''

    the _debug  method, which can be used to (optionally) write debugging messages

    '''
    # check to see if the message should be printed
    if VERBOSE:
        print("**** {} {} -".format('\033[92m' + msgType + '\033[0m', msg))

def create_hdf5_facedataset(in_images_folder_path, out_hdf5_dataset_fullPath_name, faceNet):
    '''
    
    Create hdf5 face dataset by registering each person face.
    The dataset contains the face images in opencv format, each image encodings obtained
    using faceNet, and labels/person names 
    
    Input:
        in_images_folder_path: path to a folder where each person images are stored subfolders
                        we assume that the subfolders, name represent face identity, on this path
                        contain images of a specific invidividual only,  image of a person contains only face
                        
        out_hdf5_dataset_fullName: full path of the dataset to be saved
        
        faceNet deep faceNet keras model
        
    '''
    
    person_subfolders = []

    for p in os.listdir(in_images_folder_path):
        ppath = os.path.join(in_images_folder_path, p)
        if os.path.isdir(ppath):
            person_subfolders.append(ppath)

    

    face_image_paths=[]

    for i, person_subfolder in enumerate(person_subfolders):
        for filename in os.listdir(person_subfolder):
            img_path = os.path.join(person_subfolder, filename)
            if filename.endswith('jpg'):
                face_image_paths.append(img_path)
                
                #person_name = os.path.splitext(os.path.dirname(img_path).split('/')[-1] )[0]
                #print(person_name)

    
    print("Total face images", len(face_image_paths))
    image_shape = (len(face_image_paths), 96, 96, 3)
    encod_shape = (len(face_image_paths), 128)
    
    # open hdf5 file and create dataset
    hdf5_file=h5py.File(out_hdf5_dataset_fullPath_name, mode='w')
    
    #hdf5_file.create_dataset("faces_encoding", shape=encod_shape, dtype=np.float32)
    # create resizable dataset as expected there may be any face in the training face images
    hdf5_file.create_dataset("faces_encoding", (3000, 128),
                             maxshape=(None, 128), dtype=np.float32) 
    
    dt = h5py.special_dtype(vlen=str)
    #hdf5_file.create_dataset("index_dic",  shape=(len(face_image_paths),), dtype=dt)
    hdf5_file.create_dataset("index_dic", (3000,),
                             maxshape=(None,), dtype=dt)
    
    #hdf5_file.create_dataset("face_imgs_cv", shape=image_shape,dtype=np.uint8)
    hdf5_file.create_dataset("face_imgs_cv", (3000, 96, 96, 3),
                             maxshape=(None, 96, 96, 3),dtype=np.uint8)
        
    # define throshod for facedetection and load mtcnn detector
    threshold = [0.3,0.6,0.7]
    faceDetector_mtcnn = mtcnn()
    
    # now process each person images one by one and encode them using the faceNet model
    # we will stored each encodings in face_encodings and their corresponding labels in index dictionary
    img_cnt=-1
    for image_path in face_image_paths:
        print("encoding face detected in : {}".format(image_path))

        image_cv = cv2.imread(image_path, 1)

        img_draw, face_imgs, face_bboxes, faces_landmarks = extract_draw_faces(faceDetector_mtcnn, image_cv, threshold)
        
        if len(face_imgs) == 0:
            continue
        
        img_cnt+=1
        
        lndmrks = np.array(faces_landmarks[0])
        
       
        person_name = os.path.splitext(os.path.dirname(image_path).split('/')[-1] )[0]
        
        #aligned_face_cv, aligned_face_lndmarks = normalize_faces_landmarks(out_img_size=(96,96), in_img=face_imgs[0], landmarks=lndmrks)
        aligned_face_cv, aligned_face_lndmarks = align_face(out_img_size=(96,96), in_img=image_cv, landmarks=lndmrks)
        
        f_encoding = img_to_encoding(aligned_face_cv,faceNet)
        
        print(f_encoding.shape)
        
        #im[None].shape ==>(1, 96, 96, 3)

        # writing data into dataset
        print("img_cnt",img_cnt)
        hdf5_file["face_imgs_cv"][img_cnt, ...] = aligned_face_cv[None]
        hdf5_file["faces_encoding"][img_cnt, ...] = f_encoding
        hdf5_file["index_dic"][img_cnt, ...] = person_name
        
        
        ## flip face vertical
        img_cnt+=1
        print("img_cnt",img_cnt)
        #flip_aligned_face_cv = cv2.flip(aligned_face_cv,1)
        #flip_lndmrks = np.flip(lndmrks, 1)
        print(lndmrks)
        flip_lndmrks = np.multiply(np.subtract(lndmrks, [image_cv.shape[1], 0]), [-1, 1])
        
        # after flip left and right position of eye changes
        print(flip_lndmrks)
        tmp =flip_lndmrks[0].copy()
        flip_lndmrks[0]=flip_lndmrks[1].copy()
        flip_lndmrks[1] = tmp.copy()
        
        tmp =flip_lndmrks[3].copy()
        flip_lndmrks[3]= flip_lndmrks[4].copy()
        flip_lndmrks[3] = tmp.copy()
        
        print(flip_lndmrks)
        
        
        flip_image_cv = cv2.flip(image_cv,1)
        
        flip_aligned_face_cv, _ =  align_face(out_img_size=(96,96), in_img=flip_image_cv, landmarks=flip_lndmrks)
        
        flip_f_encoding = img_to_encoding(flip_aligned_face_cv,faceNet)
        
        # writing data into dataset
        hdf5_file["face_imgs_cv"][img_cnt, ...] = flip_aligned_face_cv[None]
        hdf5_file["faces_encoding"][img_cnt, ...] = flip_f_encoding
        hdf5_file["index_dic"][img_cnt, ...] = person_name
        
        ## draw on images 
        
        face_lndmrk_draw = face_imgs[0].copy()
        
        cv2.putText(face_lndmrk_draw, '1', (lndmrks[0][0],lndmrks[0][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(face_lndmrk_draw, (lndmrks[0][0],lndmrks[0][1]), 2, (0, 255, 0))
        
        cv2.putText(face_lndmrk_draw, '2', (lndmrks[1][0], lndmrks[1][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(face_lndmrk_draw, (lndmrks[1][0],lndmrks[1][1]), 2, (0, 255, 0))
        
        cv2.putText(face_lndmrk_draw, '3', (lndmrks[2][0], lndmrks[2][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(face_lndmrk_draw, (lndmrks[2][0],lndmrks[2][1]), 2, (0, 255, 0))
        
        cv2.putText(face_lndmrk_draw, '4', (lndmrks[3][0], lndmrks[3][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(face_lndmrk_draw, (lndmrks[3][0],lndmrks[3][1]), 2, (0, 255, 0))
        
        cv2.putText(face_lndmrk_draw, '5', (lndmrks[4][0], lndmrks[4][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(face_lndmrk_draw, (lndmrks[4][0],lndmrks[4][1]), 2, (0, 255, 0))
        
        cv2.imshow("face", face_lndmrk_draw)
        
        
        aligned_face_draw = aligned_face_cv.copy()

        
        cv2.putText(aligned_face_draw, '1', (aligned_face_lndmarks[0][0],aligned_face_lndmarks[0][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(aligned_face_draw,  (aligned_face_lndmarks[0][0],aligned_face_lndmarks[0][1]), 2, (0, 255, 0))
        cv2.putText(aligned_face_draw, '2', (aligned_face_lndmarks[1][0], aligned_face_lndmarks[1][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(aligned_face_draw,  (aligned_face_lndmarks[1][0],aligned_face_lndmarks[1][1]), 2, (0, 255, 0))
        
        cv2.imshow("aligned", aligned_face_draw)
        
        
        cv2.imshow("face image", img_draw)
        
        debug("face image of {}".format(person_name))
        if( cv2.waitKey(100) == ord('q')):
            break
        
        
    
    # resize datasets according to total count img_cnt
    
    hdf5_file["faces_encoding"].resize((img_cnt, 128))
    hdf5_file["index_dic"].resize((img_cnt,))
    hdf5_file["face_imgs_cv"].resize((img_cnt, 96, 96, 3))
    
    hdf5_file.close()  
    
if __name__ == '__main__':
    
    np.set_printoptions(threshold=np.nan)
    
    debug("Initializing face Net recognition model ....")
    
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
        register persons by encoding their faces 
    '''
    debug(" Regitering faces of the person images in {} directory and save the results in {}".format(FACES_IMAGES_PATH, OUT_REG_FACES_PATH))
    #save_encoded_faces_pickle(FACES_IMAGES_PATH,OUT_REG_FACES_PATH, faceNet)
    out_hdf5_dataset_fullName = os.path.join(OUT_REG_FACES_PATH, HDF5_FACE_DATASET_NAME)
    create_hdf5_facedataset(FACES_IMAGES_PATH, out_hdf5_dataset_fullName, faceNet )
    
    