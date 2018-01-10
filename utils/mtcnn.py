'''
Created on Nov 30, 2017

@author: Inayatullah Khan
@email: inayatkh@gmail.com

In this utility module some code and ideas are taken from the Keras Implementation of Joint Face Detection and Alignment
using `Multi-task Cascaded Convolutional Neural Networks MTCCNN <https://github.com/xiangrufan/keras-mtcnn>`_.

Which is basically  transplanted from MTCNN-caffe from CongweiLin's `<https://github.com/CongWeilin/mtcnn-caffe>`_

'''


import time
import cv2

import keras.layers as KL

# from keras.layers import Conv2D, Input,MaxPool2D
# , Reshape,Activation,Flatten, Dense, Permute
from keras.models import Model
# , Sequential
# import tensorflow as tf
from keras.layers.advanced_activations import PReLU
import numpy as np


# import keras.backend as KB
# KB.set_image_data_format('channels_last')
# I have passed the data_format="channels_last" as an argument to the conv and maxpool layer


class tools(object):
    '''
     This class implements some important functions, which are mainly defined as static methods. These
     methods or functions are used by the mtcnn face detector

    '''


    def __init__(self):
        '''
        Constructor
        '''


    @staticmethod
    def rect2square(rectangles):
        '''
        Function:
            change rectangles into squares (matrix version)
        Input:
          rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
        Output:
            squares: same as input

        '''
        w = rectangles[:, 2] - rectangles[:, 0]
        h = rectangles[:, 3] - rectangles[:, 1]
        l = np.maximum(w, h).T
        rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
        rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
        rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l], 2, axis=0).T
        return rectangles

    @staticmethod
    def NMS(rectangles, threshold, type):
        '''
         Function:
             apply NMS(non-maximum suppression) on ROIs in same scale(matrix version)
        Input:
            rectangles: rectangles[i][0:3] is the position, rectangles[i][4] is score
        Output:
            rectangles: same as input

        '''

        if len(rectangles) == 0:
            return rectangles

        boxes = np.array(rectangles)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
        I = np.array(s.argsort())
        pick = []

        while len(I) > 0:
            xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])  # I[-1] have hightest prob score, I[0:-1]->others
            yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
            xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
            yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            if type == 'iom':
                o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
            else:
                o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
            pick.append(I[-1])
            I = I[np.where(o <= threshold)[0]]

        result_rectangle = boxes[pick].tolist()
        return result_rectangle

    @staticmethod
    def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
        '''
        Function:
            Detect face position and calibrate bounding box on 12net feature map(matrix version)
        Input:
            cls_prob : softmax feature map for face classify
            roi      : feature map for regression
            out_side : feature map's largest size
            scale    : current input image scale in multi-scales
            width    : image's origin width
            height   : image's origin height
            threshold: 0.6 can have 99% recall rate

        '''
        in_side = 2 * out_side + 11
        stride = 0
        if out_side != 1:
            stride = float(in_side - 12) / (out_side - 1)
        (x, y) = np.where(cls_prob >= threshold)
        boundingbox = np.array([x, y]).T
        bb1 = np.fix((stride * (boundingbox) + 0) * scale)
        bb2 = np.fix((stride * (boundingbox) + 11) * scale)
        boundingbox = np.concatenate((bb1, bb2), axis=1)
        dx1 = roi[0][x, y]
        dx2 = roi[1][x, y]
        dx3 = roi[2][x, y]
        dx4 = roi[3][x, y]
        score = np.array([cls_prob[x, y]]).T
        offset = np.array([dx1, dx2, dx3, dx4]).T
        boundingbox = boundingbox + offset * 12.0 * scale
        rectangles = np.concatenate((boundingbox, score), axis=1)
        rectangles = tools.rect2square(rectangles)
        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0     , rectangles[i][0]))
            y1 = int(max(0     , rectangles[i][1]))
            x2 = int(min(width , rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            sc = rectangles[i][4]
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, sc])

        return tools.NMS(pick, 0.3, 'iou')

    @staticmethod    
    def filter_face_24net(cls_prob, roi, rectangles, width, height, threshold):
        '''
        Function:
            Filter face position and calibrate bounding box on 12net's output
        Input:
            cls_prob  : softmax feature map for face classify
            roi_prob  : feature map for regression
            rectangles: 12net's predict
            width     : image's origin width
            height    : image's origin height
            hreshold : 0.6 can have 97% recall rate
        Output:
            rectangles: possible face positions
                 
        '''
        prob = cls_prob[:, 1]
        pick = np.where(prob >= threshold)
        rectangles = np.array(rectangles)
        x1 = rectangles[pick, 0]
        y1 = rectangles[pick, 1]
        x2 = rectangles[pick, 2]
        y2 = rectangles[pick, 3]
        sc = np.array([prob[pick]]).T
        dx1 = roi[pick, 0]
        dx2 = roi[pick, 1]
        dx3 = roi[pick, 2]
        dx4 = roi[pick, 3]
        w = x2 - x1
        h = y2 - y1
        x1 = np.array([(x1 + dx1 * w)[0]]).T
        y1 = np.array([(y1 + dx2 * h)[0]]).T
        x2 = np.array([(x2 + dx3 * w)[0]]).T
        y2 = np.array([(y2 + dx4 * h)[0]]).T
        rectangles = np.concatenate((x1, y1, x2, y2, sc), axis=1)
        rectangles = tools.rect2square(rectangles)
        pick = []
        
        for i in range(len(rectangles)):
            x1 = int(max(0     , rectangles[i][0]))
            y1 = int(max(0     , rectangles[i][1]))
            x2 = int(min(width , rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            sc = rectangles[i][4]
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, sc])
        return tools.NMS(pick, 0.3, 'iou')
    
    
    @staticmethod
    def filter_face_48net(cls_prob, roi, pts, rectangles, width, height, threshold):
        '''
        Function:
            Filter face position and calibrate bounding box on 12net's output
        Input:
            cls_prob  : cls_prob[1] is face possibility
            roi       : roi offset
            pts       : 5 landmark
            rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
            width     : image's origin width
            height    : image's origin height
            threshold : 0.7 can have 94% recall rate on CelebA-database
        Output:
            rectangles: face positions and landmarks
        '''
        prob = cls_prob[:, 1]
        pick = np.where(prob >= threshold)
        rectangles = np.array(rectangles)
        x1 = rectangles[pick, 0]
        y1 = rectangles[pick, 1]
        x2 = rectangles[pick, 2]
        y2 = rectangles[pick, 3]
        sc = np.array([prob[pick]]).T
        dx1 = roi[pick, 0]
        dx2 = roi[pick, 1]
        dx3 = roi[pick, 2]
        dx4 = roi[pick, 3]
        w = x2 - x1
        h = y2 - y1
        pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
        pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
        pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
        pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
        pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
        pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
        pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
        pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
        pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
        pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T
        # pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
        # pts1 = np.array([(h * pts[pick, 1] + y1)[0]]).T
        # pts2 = np.array([(w * pts[pick, 2] + x1)[0]]).T
        # pts3 = np.array([(h * pts[pick, 3] + y1)[0]]).T
        # pts4 = np.array([(w * pts[pick, 4] + x1)[0]]).T
        # pts5 = np.array([(h * pts[pick, 5] + y1)[0]]).T
        # pts6 = np.array([(w * pts[pick, 6] + x1)[0]]).T
        # pts7 = np.array([(h * pts[pick, 7] + y1)[0]]).T
        # pts8 = np.array([(w * pts[pick, 8] + x1)[0]]).T
        # pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T
        x1 = np.array([(x1 + dx1 * w)[0]]).T
        y1 = np.array([(y1 + dx2 * h)[0]]).T
        x2 = np.array([(x2 + dx3 * w)[0]]).T
        y2 = np.array([(y2 + dx4 * h)[0]]).T
        rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9), axis=1)
        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0     , rectangles[i][0]))
            y1 = int(max(0     , rectangles[i][1]))
            x2 = int(min(width , rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            # print("---------------------",x1,x2,y1,y2)
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, rectangles[i][4],
                             rectangles[i][5], rectangles[i][6],
                             rectangles[i][7], rectangles[i][8],
                             rectangles[i][9], rectangles[i][10],
                             rectangles[i][11], rectangles[i][12],
                             rectangles[i][13], rectangles[i][14]])
        return tools.NMS(pick, 0.3, 'iom')
    
    @staticmethod
    def calculateScales(img):
        '''
        
        Function:
            calculate multi-scale and limit the maxinum side to 1000
        Input:
            img: original image
        Output:
            pr_scale: limit the maxinum side to 1000, < 1.0
            scales  : Multi-scale
        
        '''
        caffe_img = img.copy()
        pr_scale = 1.0
        h, w, ch = caffe_img.shape
        if min(w, h) > 500:
            pr_scale = 500.0 / min(h, w)
            w = int(w * pr_scale)
            h = int(h * pr_scale)
        elif max(w, h) < 500:
            pr_scale = 500.0 / max(h, w)
            w = int(w * pr_scale)
            h = int(h * pr_scale)
            
        # multi-scale
        scales = []
        factor = 0.709
        factor_count = 0
        minl = min(h, w)
        while minl >= 12:
            scales.append(pr_scale * pow(factor, factor_count))
            minl *= factor
            factor_count += 1
        return scales
    



    @staticmethod
    def filter_face_48net_newdef(cls_prob, roi, pts, rectangles, width, height, threshold):
        '''
        
        Function:
            calculate   landmark point , new def
        Input:
            cls_prob  : cls_prob[1] is face possibility
            roi       : roi offset
            pts       : 5 landmark
            rectangles: 12net's predict, rectangles[i][0:3] is the position, rectangles[i][4] is score
            width     : image's origin width
            height    : image's origin height
            threshold : 0.7 can have 94% recall rate on CelebA-database
        Output:
            rectangles: face positions and landmarks
            
        '''
    
        prob = cls_prob[:, 1]
        pick = np.where(prob >= threshold)
        rectangles = np.array(rectangles)
        x1 = rectangles[pick, 0]
        y1 = rectangles[pick, 1]
        x2 = rectangles[pick, 2]
        y2 = rectangles[pick, 3]
        sc = np.array([prob[pick]]).T
        dx1 = roi[pick, 0]
        dx2 = roi[pick, 1]
        dx3 = roi[pick, 2]
        dx4 = roi[pick, 3]
        w = x2 - x1
        h = y2 - y1
        pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
        pts1 = np.array([(h * pts[pick, 1] + y1)[0]]).T
        pts2 = np.array([(w * pts[pick, 2] + x1)[0]]).T
        pts3 = np.array([(h * pts[pick, 3] + y1)[0]]).T
        pts4 = np.array([(w * pts[pick, 4] + x1)[0]]).T
        pts5 = np.array([(h * pts[pick, 5] + y1)[0]]).T
        pts6 = np.array([(w * pts[pick, 6] + x1)[0]]).T
        pts7 = np.array([(h * pts[pick, 7] + y1)[0]]).T
        pts8 = np.array([(w * pts[pick, 8] + x1)[0]]).T
        pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T
        x1 = np.array([(x1 + dx1 * w)[0]]).T
        y1 = np.array([(y1 + dx2 * h)[0]]).T
        x2 = np.array([(x2 + dx3 * w)[0]]).T
        y2 = np.array([(y2 + dx4 * h)[0]]).T
        rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9), axis=1)
        # print (pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9)
        pick = []
        for i in range(len(rectangles)):
            x1 = int(max(0     , rectangles[i][0]))
            y1 = int(max(0     , rectangles[i][1]))
            x2 = int(min(width , rectangles[i][2]))
            y2 = int(min(height, rectangles[i][3]))
            if x2 > x1 and y2 > y1:
                pick.append([x1, y1, x2, y2, rectangles[i][4],
                     rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9], rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
        return tools.NMS(pick, 0.3, 'idsom')


    @staticmethod
    def imglist_meanvalue(img_nparray):
        '''
        Function:
            calculate mean value of img_list for double checck img quality
        Input:
            img_nparray: numpy array of input
        Output:
            img_nparray: numpy array of img mean value
        '''
        img_mean_array = np.mean(img_nparray , axis=(1, 2, 3))
        return np.array(img_mean_array)

    
    

class mtcnn(object):
    '''
    
    This class implements the MTCNN face detector
    
    '''
    
    def __init__(self, kao_pnet_weight_path='./models/mtcnn/12net.h5', kao_rnet_weight_path='./models/mtcnn/24net.h5', kao_onet_weight_path='./models/mtcnn/48net.h5'):
        '''
            Constructor
        '''
        
        self.Pnet = self._create_Kao_Pnet(weight_path=kao_pnet_weight_path)
        self.Rnet = self._create_Kao_Rnet(weight_path=kao_rnet_weight_path)
        self.Onet = self._create_Kao_Onet(weight_path=kao_onet_weight_path) 
    
    def detectFace(self, img, threshold):
        '''
        
        '''
        caffe_img = (img.copy() - 127.5) / 127.5
        
        origin_h, origin_w, ch = caffe_img.shape
        scales = tools.calculateScales(img)
        out = []
        # t0 = time.time()
        # # del scales[:4]
    
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            scale_img = cv2.resize(caffe_img, (ws, hs))
            input = scale_img.reshape(1, *scale_img.shape)
            ouput = self.Pnet.predict(input)  # .transpose(0,2,1,3) should add, but seems after process is wrong then.
            out.append(ouput)
        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            cls_prob = out[i][0][0][:, :,
                       1]  # i = #scale, first 0 select cls score, second 0 = batchnum, alway=0. 1 one hot repr
            roi = out[i][1][0]
            out_h, out_w = cls_prob.shape
            out_side = max(out_h, out_w)
            # print('calculating img scale #:', i)
            cls_prob = np.swapaxes(cls_prob, 0, 1)
            roi = np.swapaxes(roi, 0, 2)
            rectangle = tools.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)
        rectangles = tools.NMS(rectangles, 0.7, 'iou')
    
        # t1 = time.time()
        # print ('time for 12 net is: ', t1-t0)
    
        if len(rectangles) == 0:
            return rectangles
    
        crop_number = 0
        out = []
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (24, 24))
            predict_24_batch.append(scale_img)
            crop_number += 1
    
        predict_24_batch = np.array(predict_24_batch)
    
        out = self.Rnet.predict(predict_24_batch)
    
        cls_prob = out[0]  # first 0 is to select cls, second batch number, always =0
        cls_prob = np.array(cls_prob)  # convert to numpy
        roi_prob = out[1]  # first 0 is to select roi, second batch number, always =0
        roi_prob = np.array(roi_prob)
        rectangles = tools.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        # t2 = time.time()
        # print ('time for 24 net is: ', t2-t1)
    
    
        if len(rectangles) == 0:
            return rectangles
    
    
        crop_number = 0
        predict_batch = []
        for rectangle in rectangles:
            # print('calculating net 48 crop_number:', crop_number)
            crop_img = caffe_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)
            crop_number += 1
    
        predict_batch = np.array(predict_batch)
    
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]  # index
        # rectangles = tools.filter_face_48net_newdef(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h,
        #                                             threshold[2])
        rectangles = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        # t3 = time.time()
        # print ('time for 48 net is: ', t3-t2)
    
        return rectangles

        
    def _create_Kao_Onet(self, weight_path='./models/mtcnn/48net.h5'):
        '''
        
        '''
        input = KL.Input(shape=[48, 48, 3])
        
        x = KL.Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1', data_format="channels_last")(input)
        x = KL.PReLU(shared_axes=[1, 2], name='prelu1')(x)
        x = KL.MaxPool2D(pool_size=3, strides=2, padding='same', data_format="channels_last")(x)
        x = KL.Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2', data_format="channels_last")(x)
        x = KL.PReLU(shared_axes=[1, 2], name='prelu2')(x)
        x = KL.MaxPool2D(pool_size=3, strides=2, data_format="channels_last")(x)
        x = KL.Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3', data_format="channels_last")(x)
        x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
        x = KL.MaxPool2D(pool_size=2, data_format="channels_last")(x)
        x = KL.Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4', data_format="channels_last")(x)
        x = KL.PReLU(shared_axes=[1, 2], name='prelu4')(x)
        x = KL.Permute((3, 2, 1))(x)
        x = KL.Flatten()(x)
        x = KL.Dense(256, name='conv5') (x)
        x = KL.PReLU(name='prelu5')(x)
    
        classifier = KL.Dense(2, activation='softmax', name='conv6-1')(x)
        bbox_regress = KL.Dense(4, name='conv6-2')(x)
        landmark_regress = KL.Dense(10, name='conv6-3')(x)
        model = Model([input], [classifier, bbox_regress, landmark_regress])
        model.load_weights(weight_path, by_name=True)
    
        return model


    def _create_Kao_Rnet(self, weight_path='./models/mtcnn/24net.h5'):
        '''
        
        '''
        input = KL.Input(shape=[24, 24, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
        x = KL.Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1', data_format="channels_last")(input)
        x = KL.PReLU(shared_axes=[1, 2], name='prelu1')(x)
        x = KL.MaxPool2D(pool_size=3, strides=2, padding='same', data_format="channels_last")(x)
    
        x = KL.Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2', data_format="channels_last")(x)
        x = KL.PReLU(shared_axes=[1, 2], name='prelu2')(x)
        x = KL.MaxPool2D(pool_size=3, strides=2, data_format="channels_last")(x)
    
        x = KL.Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3', data_format="channels_last")(x)
        x = KL.PReLU(shared_axes=[1, 2], name='prelu3')(x)
        x = KL.Permute((3, 2, 1))(x)
        x = KL.Flatten()(x)
        x = KL.Dense(128, name='conv4')(x)
        x = KL.PReLU(name='prelu4')(x)
        classifier = KL.Dense(2, activation='softmax', name='conv5-1')(x)
        bbox_regress = KL.Dense(4, name='conv5-2')(x)
        model = Model([input], [classifier, bbox_regress])
        model.load_weights(weight_path, by_name=True)
        return model


    def _create_Kao_Pnet(self, weight_path='./models/mtcnn/12net.h5'):
        '''
        
        '''
        
        input = KL.Input(shape=[None, None, 3])
        
        x = KL.Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1', data_format="channels_last")(input)
        x = KL.PReLU(shared_axes=[1, 2], name='PReLU1')(x)
        x = KL.MaxPool2D(pool_size=2, data_format="channels_last")(x)
        x = KL.Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2', data_format="channels_last")(x)
        x = KL.PReLU(shared_axes=[1, 2], name='PReLU2')(x)
        x = KL.Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3', data_format="channels_last")(x)
        x = KL.PReLU(shared_axes=[1, 2], name='PReLU3')(x)
        classifier = KL.Conv2D(2, (1, 1), activation='softmax', name='conv4-1', data_format="channels_last")(x)
        bbox_regress = KL.Conv2D(4, (1, 1), name='conv4-2', data_format="channels_last")(x)
        model = Model([input], [classifier, bbox_regress])
        model.load_weights(weight_path, by_name=True)
        return model
