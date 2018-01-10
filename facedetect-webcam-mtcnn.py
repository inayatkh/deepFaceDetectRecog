'''
Created on Sep 9, 2017
@author: inayat
'''

# import the required  packages
from imutils.video import WebcamVideoStream
#from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2




from utils import FPS2

from utils import mtcnn

from utils import extract_draw_faces
from utils import align_face


if __name__ == '__main__':
    
   
    
       
    
    print("[info] starting to read a webcam ...")
    capWebCam = WebcamVideoStream(0).start()
    time.sleep(1.0)
    
    # start the frame per second  (FPS) counter
    fps = FPS2().start() 
    
    
    #threshold = [0.6,0.6,0.7]
    threshold = [0.3,0.6,0.7]
    faceDetector = mtcnn()
    
    dispWin = "Deep Face Detection (MTCNN)"
    #cv2.namedWindow(dispWin, cv2.WINDOW_NORMAL)
    cv2.namedWindow(dispWin, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO )
    
    
    # loop over the frames obtained from the webcam
    while True:
        # grab each frame from the threaded  stream,
        # resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        frame1 = capWebCam.read()
        frame = cv2.flip(frame1,1)
        #frame = imutils.resize(frame, width=450)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = np.dstack([frame, frame, frame])
        
        # display the size of the queue on the frame
        #cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        frame_draw, face_imgs, face_bboxes, faces_landmarks = extract_draw_faces(faceDetector,frame, threshold)
        
        if len(face_imgs) == 0:
            continue
        
        '''
        if len(face_imgs) != 0:
              face_montages= imutils.build_montages(face_imgs, image_shape=(96,96), montage_shape=(len(face_imgs), 1))
              for i, face  in enumerate(face_montages):
                  #print(i)
                  cv2.imshow("faces montages " + str(i) , face)
        '''
        
        
        
        ## draw one of the normalized face
        face_lndmrk_draw = face_imgs[0].copy()
        
        
        lndmrks = np.array(faces_landmarks[0])
       
        '''
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
        '''
        
        cv2.imshow("orig", face_lndmrk_draw)
        
        #face_cv =  cv2.resize(face_imgs[0], (96,96), interpolation=cv2.INTER_CUBIC)
        
        #cv2.imshow("face 96x96", face_cv)
        #aligned_face, aligned_face_lndmarks = normalize_faces_landmarks(out_img_size=(96,96), in_img=face_imgs[0], landmarks=lndmrks)
        #aligned_face, aligned_face_lndmarks = align_face(out_img_size=(96,96), in_img=face_imgs[0], landmarks=lndmrks)
        
        aligned_face, aligned_face_lndmarks = align_face(out_img_size=(96,96), in_img=frame, landmarks=lndmrks)
        
        cv2.putText(aligned_face, '1', (aligned_face_lndmarks[0][0],aligned_face_lndmarks[0][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(aligned_face,  (aligned_face_lndmarks[0][0],aligned_face_lndmarks[0][1]), 2, (0, 255, 0))
        cv2.putText(aligned_face, '2', (aligned_face_lndmarks[1][0], aligned_face_lndmarks[1][1]), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
        cv2.circle(aligned_face,  (aligned_face_lndmarks[1][0],aligned_face_lndmarks[1][1]), 2, (0, 255, 0))
        
        cv2.imshow("aligned", aligned_face)
        
        
        
        fps.update()
        cv2.putText(frame_draw, "FPS: {:.2f}".format(fps.fps()),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        
        # show the frame and update the FPS counter
        cv2.imshow(dispWin, frame)
        
        
        
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        
        
        
        
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    capWebCam.stop()
