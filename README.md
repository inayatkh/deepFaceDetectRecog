
# Deep Face Detection and Recognition
___________________________________________

## Introduction 

Here, I have demonstrated the use of Multi-task Cascaded Convolutional Neural Networks, [MTCCNN](https://github.com/xiangrufan/keras-mtcnn)  for deep face detection and [faceNet](https://arxiv.org/pdf/1503.03832.pdf) for encoding of the aligned faces. Recognition is then performed by matching the faces detected in the test images with the already enrolled or registered faces using euclidean distances.

Due to time limit, I have not included the detailed document.  I will later include the more detailed documentation to this project.

## Requirements

- python version 3.5
- tensor flow
- keras
- hdf5
- some othe packages

note: full requirement i will update in near future

Using this code, you can build your own  face recognition system just by adding the images of faces of the persons need to be recognised in the "faces" directory inside the "images" folder.  The following steps must then be followed;



## Step 1: Register Faces

The faces to be recognised must be registered and enrolled in the system. This is done by executing the following command

$ python registerFaces.py

This will detected the faces in the training images and their corresponding encoding are saved in a dataset.

## Step 2 :
Firstly, excute the following code to recognise faces in the test images

$ python recognisePersons.py

