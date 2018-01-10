'''
Created on Dec 18, 2017

@author: inayat
'''

# switch off warnings
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from utils import mtcnn
from utils import align_face
from utils import build_montages


import cv2

import numpy as np


VERBOSE = True
#import datetime

def debug(msg, msgType="[INFO]"):
    '''
      the _debug  method, which can be used to (optionally) write debugging messages
    '''
    # check to see if the message should be printed
    if VERBOSE:
        print("**** {} {} -".format('\033[92m' + msgType + '\033[0m', msg))
        
        
if __name__ == '__main__':
    
    
    np.set_printoptions(threshold=np.nan)
    
    
    '''
    MTCNN face detection
    
    '''
    
    
    
    #threshold = [0.6,0.6,0.7]
    
    debug("Loading Deep face Detector (MTCNN) recognition model ....")
    threshold = [0.3,0.6,0.7]
    faceDetector = mtcnn()
    
    
    
    debug("Loading Registered Persons faces encodings and index files generated with registerFaces.py ....")
    
   
    
   
    test_images_path = './images/test3'
    #test_images_path = './images/faces/basit'
    #test_images_path = './images/test2'
    test_images_result_path = './images/result-test3'
    #test_images_result_path = './images/result-test-2'
    
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
            
            #face_id =0;
            aligned_faces_list =[]
            
            
            
            for bbox_lndmarks in bboxes_lndmarks:

                if bbox_lndmarks is not None:

                    left, top,  right, bottom = bbox_lndmarks[0], bbox_lndmarks[1], bbox_lndmarks[2], bbox_lndmarks[3]
                    
                    left =int(left)
                    top = int(top)
                    right = int(right)
                    bottom = int(bottom)
                    
                  

                    #face_bboxes.append([top, left, bottom, right])
                    test_face_img = test_img[top:bottom, left:right]
                    
                    print(top, bottom, left, right)
                    print(" test_face_img ", test_face_img.shape)
                    test_face_lndmarks =[]
                    for i in range(5, 15, 2):
                        #test_face_lndmarks.append([int(bbox_lndmarks[i + 0] - left), int(bbox_lndmarks[i + 1]-top)])
                        test_face_lndmarks.append([int(bbox_lndmarks[i + 0]), int(bbox_lndmarks[i + 1])])
                        
                    
                    test_face_lndmarks = np.array(test_face_lndmarks)
                    
                    ### encoding test_face
                    
                    #test_face_img = cv2.resize(test_face_img, (96,96), interpolation=cv2.INTER_CUBIC)
                    '''
                    normalized_face_img, normalized_face_lndmarks = normalize_faces_landmarks(out_img_size=(96,96),
                                                                                              in_img=test_face_img,
                                                                                              landmarks=test_face_lndmarks)
                    '''
                   
                    a_face_img, _tmp =align_face(out_img_size=(96,96),in_img=test_img, landmarks=test_face_lndmarks)
                    
                    aligned_faces_list.append(a_face_img)
                    
                    
                    
                    ######## drawing faces and labels or person names
                    
                    #face_id +=1
                    #cv2.imshow("Aligned Face " + str(face_id), normalized_face_img)
                    DRAW_COLOR = (0, 255, 0)
                        
                    start_pt = (int(bbox_lndmarks[0]), int(bbox_lndmarks[1]))
                    end_pt = (int(bbox_lndmarks[2]), int(bbox_lndmarks[3]))
                    for i in range(thickness):
                        start_pt = (int(bbox_lndmarks[0])+i, int(bbox_lndmarks[1])+i)
                        end_pt = (int(bbox_lndmarks[2])-i, int(bbox_lndmarks[3])-i)
                        cv2.rectangle(test_img_draw, start_pt, end_pt, DRAW_COLOR, 1)
                        
                    for i in range(5, 15, 2):
                        cv2.circle(test_img_draw, (int(bbox_lndmarks[i + 0]), int(bbox_lndmarks[i + 1])), 2, (0, 255, 0))
                    
                    
                    #person_id_tag = "%s: %.2f" % (identity, dist)
                                        
                    #text_end_pt   = (int(bbox_lndmarks[0]) +  len(person_id_tag) * 10, int(bbox_lndmarks[1]) - 20 )
                    #text_start_pt = (max(int(bbox_lndmarks[0]), 10), max(int(bbox_lndmarks[1]), 10))
                    
                    #cv2.rectangle(test_img_draw,text_start_pt , text_end_pt,
                    #              DRAW_COLOR, -1, cv2.LINE_AA)
                    #cv2.putText(test_img_draw, person_id_tag, text_start_pt, cv2.FONT_ITALIC,0.4, (255, 255, 255), 1)
            
            
            if len(aligned_faces_list) > 0 :
                
                montage_shape = (len(aligned_faces_list),1)
                
                faces_montages = build_montages(aligned_faces_list, image_shape=(96,96),
                                                montage_shape=montage_shape)
                for num, faces_montage in enumerate(faces_montages):
                    cv2.imshow("Aligned Faces" + "  " + str(num), faces_montage)
                    #cv2.imwrite("./images/" + name + "-" + str(num) + ".jpg" , faces_montage)
            
                
            
            
            
        
            cv2.imshow(dispWin, test_img_draw)
            #test_img_file_path = os.path.join(test_images_result_path, img_file_name)
            #cv2.imwrite(test_img_file_path, test_img_draw)
            if cv2.waitKey(0) == ord('q'):
                break
            
                
            
    
    cv2.destroyAllWindows()