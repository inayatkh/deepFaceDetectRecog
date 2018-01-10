'''
Created on Nov 28, 2017

@author: Inayatullah Khan
@email: inayatkh@gmail.com

In this module the necessary function related to model training  are defined.
'''

from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as KB
KB.set_image_data_format('channels_first')
import cv2
import math
import numpy as np
from numpy import genfromtxt

import tensorflow as tf



def extract_draw_faces(faceDetector, img, threshold=[0.3,0.6,0.7]):
    '''

     implementation of drawing bounding boxes around detected faces using the mtcnn detector
     and an important role is the extraction of the face plus some space around it, which is then used 
     to be normalized or aligned into 96x96. This basically, remove any black pixel in the then normalized or aligend faces.
     

    '''

    thickness = (img.shape[0] + img.shape[1]) // 300

    img_draw = img.copy()

    bboxes = faceDetector.detectFace(img, threshold)

    face_bboxes = []
    landmarks_face_img = []
    face_imgs = []

    for bbox in bboxes:
        if bbox is not None:


            top, left, bottom, right = bbox[1], bbox[0], bbox[3], bbox[2]
            
            left =int(left)
            top = int(top)
            right = int(right)
            bottom = int(bottom)
            
            '''
            #face_bbox_margin = 0.5
            face_bbox_extra_margin = (right - left)/4

            #top = max(0, np.floor(top + face_bbox_margin).astype('int32'))
            #left = max(0, np.floor(left + face_bbox_margin).astype('int32'))
            
            top = max(0, np.floor(top - face_bbox_extra_margin).astype('int32'))
            left = max(0, np.floor(left - face_bbox_extra_margin).astype('int32'))
            
            bottom = min(img.shape[0], np.floor(bottom + face_bbox_extra_margin).astype('int32'))
            right = min(img.shape[1], np.floor(right + face_bbox_extra_margin).astype('int32'))
            
            '''

            face_bboxes.append([top, left, bottom, right])
            face_img = img[top:bottom, left:right]
            face_imgs.append(face_img)

            start_pt = (left, top)
            end_pt = (right, bottom)
            for i in range(thickness):
                start_pt = (left+i, top+i)
                end_pt = (right-i, bottom-i)
                cv2.rectangle(img_draw, start_pt, end_pt, (255, 255, 0), 1)
            '''
            W = -int(bbox[0]) + int(bbox[2])
            H = -int(bbox[1]) + int(bbox[3])
            paddingH = 0.01 * W
            paddingW = 0.02 * H
            
            crop_img = img[int(bbox[1]+paddingH):int(bbox[3]-paddingH),
                           int(bbox[0]-paddingW):int(bbox[2]+paddingW)]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            if crop_img is None:
                continue
            if crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                continue
            '''
            cv2.rectangle(img_draw, (int(bbox[0]), int(bbox[1])),
                           (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)


            landmarks = []
            for i in range(5, 15, 2):
                cv2.circle(img_draw, (int(bbox[i + 0]), int(bbox[i + 1])), 2, (0, 255, 0))
                #landmarks.append([int(bbox[i + 0] - left), int(bbox[i + 1]-top)])
                landmarks.append([int(bbox[i + 0]), int(bbox[i + 1])])
            
            landmarks_face_img.append(landmarks)
            
            cv2.putText(img_draw, '1', (int(bbox[5]), int(bbox[6])), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
            cv2.putText(img_draw, '2', (int(bbox[7]), int(bbox[8])), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
            cv2.putText(img_draw, '3', (int(bbox[9]), int(bbox[10])), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
            cv2.putText(img_draw, '4', (int(bbox[11]), int(bbox[12])), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
            cv2.putText(img_draw, '5', (int(bbox[13]), int(bbox[14])), cv2.FONT_ITALIC,0.4, (0,0,255), 1)
            
            
            


    return (img_draw, face_imgs, face_bboxes, landmarks_face_img)


def align_face(out_img_size, in_img, landmarks):
        '''
            Implementation of facial Alignment, face is aligned
            such that it is 
            1) Centered in the out image
            2) Eyes lis on a horizontal line ( eyes are rotated
            such that the eyes lie along the same y-coord
            3) Face is scaled such that the size of faces are approx
            identicall
        
        '''
        

        # compute the center of mass for each eye
        # Center landmark pts of the eyes in the input image face image in_img
        leftEyeCenter = landmarks[1]
        rightEyeCenter = landmarks[0]

        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        #desiredLeftEye = (0.35, 0.35)
        #desiredLeftEye = (0.3, 0.3)
        desiredLeftEye = (0.2, 0.15)
        desiredFaceWidth = out_img_size[0]
        desiredFaceHeight = out_img_size[0]
        
        desiredRightEyeX = 1.0 - desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        #tX = desiredFaceWidth * 0.5
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        output = cv2.warpAffine(in_img, M, (w, h),flags=cv2.INTER_CUBIC)

        # return the aligned face
        
        # optional apply similarity transform to the out landmarks
        out_pts = np.reshape(landmarks, (landmarks.shape[0], 1, landmarks.shape[1]))
    
        out_landmarks = cv2.transform(out_pts, M)
    
        out_landmarks = np.reshape(out_landmarks, (landmarks.shape[0], landmarks.shape[1]))
        
        return output, out_landmarks
    
def loss_triplet(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss function, details are explained bellow

    .. math::

        \\mathcal{J} = \\sum^{m}_{i=1} \\large[ \\small \\underbrace{\\mid \\mid f(A^{(i)}) - f(P^{(i)}) \\mid \\mid_2^2}_\\text{(1)} - \\underbrace{\\mid \\mid f(A^{(i)}) - f(N^{(i)}) \\mid \\mid_2^2}_\\text{(2)} + \\alpha \\large ] \\small_+
    Here, the notation :math:`[z]_+`  is used to denote :math:`max(z,0)`

    We want the term (1) to be small where it is the distance between the anchor image encodings "A" and 
    the positive "P" for a given triplet.

    We want the term (2) to be relatively large, where it is the squared distance between the anchor "A"
    and the negative "N" for a given triplet.

    :math:`\\alpha` is  the hyperparameter called the margin which should be picked manually.
    The dafault value is  :math:`\\alpha=0.2`

    Arguments:

    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.

    y_pred -- python list containing three objects:

            anchor -- the encodings for the anchor images, of shape (None, 128)

            positive -- the encodings for the positive images, of shape (None, 128)

            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:

    loss -- real number, value of the loss

    """

    y_true = y_true

    y_true = None

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # step 1 compute the encoding distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)

    # Step 2: Compute the (encoding) distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0))

    print("loss_triplet", loss)

    return loss

def build_montages(image_list, image_shape, montage_shape):
    """
    This implementations are transformed from imutils package, with a little modification. We have
    converted the background of the montage from black into white 
    ---------------------------------------------------------------------------------------------
    author: Kyle Hounslow
    ---------------------------------------------------------------------------------------------
    Converts a list of single images into a list of 'montage' images of specified rows and columns.
    A new montage image is started once rows and columns of montage image is filled.
    Empty space of incomplete montage images are filled with black pixels
    ---------------------------------------------------------------------------------------------
    :param image_list: python list of input images
    :param image_shape: tuple, size each image will be resized to for display (width, height)
    :param montage_shape: tuple, shape of image montage (width, height)
    :return: list of montage images in numpy array format
    ---------------------------------------------------------------------------------------------

    example usage:

    # load single image
    img = cv2.imread('lena.jpg')
    # duplicate image 25 times
    num_imgs = 25
    img_list = []
    for i in xrange(num_imgs):
        img_list.append(img)
    # convert image list into a montage of 256x256 images tiled in a 5x5 montage
    montages = make_montages_of_images(img_list, (256, 256), (5, 5))
    # iterate through montages and display
    for montage in montages:
        cv2.imshow('montage image', montage)
        cv2.waitKey(0)

    ----------------------------------------------------------------------------------------------
    """
    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')
    image_montages = []
    # start with black canvas to draw images onto
    #montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
    #                      dtype=np.uint8)
    # start with white canvas to draw images onto
    montage_image = np.ones(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                          dtype=np.uint8) * 255
    cursor_pos = [0, 0]
    start_new_img = False
    for img in image_list:
        if type(img).__module__ != np.__name__:
            raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = cv2.resize(img, image_shape)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                #montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                #                      dtype=np.uint8)
                # reset white canvas
                montage_image = np.ones(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                                      dtype=np.uint8) *255
                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    return image_montages
