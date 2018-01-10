'''
Created on Dec 12, 2017

@author: Inayatullah Khna
@email: inayatkh@gmail.com


'''
from utils import build_montages

import os
import h5py
import cv2
import numpy as np
import math

import itertools
import matplotlib.pyplot as plt

from matplotlib import colors

from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.ticker as ticker

import seaborn as sb 
import pandas as pd
REG_FACES_PATH = "./images/registeredfaces"
HDF5_FACE_DATASET_NAME = "registeredFaces.h5"

def get_distances_matrix(reg_faces_encodings, index_dic):
    '''
      Calculate inter faces euclidean distances among tegistered aligned faces
      
      input:
          reg_faces_encodings : 128 dim face encodings, whoes shape is (no_faces, 128)
          index_dic: person name dictionary whose size = reg_faces_encodings.shape[0]
    
    '''
    
    num_faces = reg_faces_encodings.shape[0]
    
    dist_matrix = np.zeros((num_faces, num_faces), dtype=np.float32)
    
    for idx, reg_face_encoding in enumerate(reg_faces_encodings):
        dist_matrix[idx,:] = np.linalg.norm(reg_faces_encodings[idx][:] - reg_faces_encodings , axis=1, keepdims=True).T
        
    return dist_matrix
def plot_dist_seaborn(dist_matrix, index_dic,face_name_index_map, title='Euclidean Distance Matrix'):
    
    '''
    
    '''
    #sb.heatmap(dist, cmap='PuOr', linewidths=0.1, linecolor='yellow', xticklabels=4)
    
    xticks = []
    xticklabels =[]
    
    for name in face_name_index_map:
        img_indx_list = face_name_index_map[name]
        xticks.append(img_indx_list[-1])
        xticklabels.append(name + " " + str(img_indx_list[-1]))
        
    '''
    df = pd.DataFrame(dist_matrix, columns=xticks)
    dpi =200
    fig= plt.figure(num=None, figsize=(800/dpi, 800/dpi), dpi=dpi, facecolor='w', edgecolor='k')
    '''
    '''
    sns.axes_style()
    {'axes.axisbelow': True,
 'axes.edgecolor': '.8',
 'axes.facecolor': 'white',
 'axes.grid': True,
 'axes.labelcolor': '.15',
 'axes.linewidth': 1.0,
 'figure.facecolor': 'white',
 'font.family': [u'sans-serif'],
 'font.sans-serif': [u'Arial',
  u'DejaVu Sans',
  u'Liberation Sans',
  u'Bitstream Vera Sans',
  u'sans-serif'],
 'grid.color': '.8',
 'grid.linestyle': u'-',
 'image.cmap': u'rocket',
 'legend.frameon': False,
 'legend.numpoints': 1,
 'legend.scatterpoints': 1,
 'lines.solid_capstyle': u'round',
 'text.color': '.15',
 'xtick.color': '.15',
 'xtick.direction': u'out',
 'xtick.major.size': 0.0,
 'xtick.minor.size': 0.0,
 'ytick.color': '.15',
 'ytick.direction': u'out',
 'ytick.major.size': 0.0,
 'ytick.minor.size': 0.0}
    
    '''
    
    sb.set(rc={"figure.figsize": (6, 6)})
    #sb.set_context("paper") 
    #sb.set_context("poster")
    #sb.set_style("whitegrid")
    sb.set_style("ticks",{'xtick.direction': u'out', 'xtick.major.size': '12.0', 'ytick.major.size': '12.0'})
    
    
    
    ax=sb.heatmap(dist_matrix, cmap='PuOr')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(xticks)
    ax.set_yticklabels(xticklabels)
    
    
    #print(dir(ax))
    figure = ax.get_figure()    
    figure.savefig("images/registeredfaces/dist_matrix.png", dpi=400)
    
    plt.show()
    
    
    
    
    #####################
    '''
    dpi =200
    fig2= plt.figure(num=None, figsize=(800/dpi, 800/dpi), dpi=dpi, facecolor='w', edgecolor='k')
    
    
    
    
    # Can be great to plot only a half matrix
    mask = np.zeros_like(dist_matrix)
    mask[np.triu_indices_from(mask)] = True
    with sb.axes_style("white"):
        p2 = sb.heatmap(dist_matrix, mask=mask, square=True, cmap='PuOr')
        
    plt.show()
    '''

    
def plot_dist_matrix(dist_matrix, index_dic,face_name_index_map, title='Euclidean Distance Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the euclidian distance matrix among the registered faces
    
    """
    
    xticks = []
    xticklabels =[]
    
    for name in face_name_index_map:
        img_indx_list = face_name_index_map[name]
        xticks.append(img_indx_list[-1])
        xticklabels.append(name + " " + str(img_indx_list[-1]))
    
    
    dpi =200
    fig= plt.figure(num=None, figsize=(800/dpi, 800/dpi), dpi=dpi, facecolor='w', edgecolor='k')
    
    ax = fig.add_subplot(111)
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    
    ax.set_yticks(xticks)
    ax.set_yticklabels(xticklabels)
    
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    
    '''
    def format_fn(tick_val, tick_pos):
        if int(tick_val) in tick_marks:
            return index_dic[int(tick_val)]
        else:
            return ''
    
    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.yaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    '''
    

    #plt.imshow(dist_matrix, interpolation='nearest', cmap=cmap)
    
    #plt.imshow(dist_matrix, interpolation='nearest', cmap='Dark2')
    
    
    bounds=[0,0.05,0.1,0.2,0.3,0.4, 0.5, 0.55, 0.60, 0.65, 0.7, 0.8, 0.85, 0.90, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    #color_list = plt.cm.Set2(np.linspace(0, 1, 4))
    #color_list = plt.cm.hot(np.linspace(0, 1, 4))
    color_list = plt.cm.Blues(np.linspace(0, 1, len(bounds)+1))
    #color_list = plt.cm.gist_heat(np.linspace(0, 1, len(bounds)+1))
    cmap =colors.ListedColormap(color_list)
    
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    img1=plt.imshow(dist_matrix, interpolation='nearest', cmap=cmap, norm=norm)
    plt.colorbar(img1, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds)
    
    
    #plt.imshow(dist_matrix, interpolation='None', cmap=cmap)
    
    plt.title(title)
    #plt.colorbar()
    
    tick_marks = np.arange(len(index_dic))
    
    
    #plt.xticks(tick_marks, index_dic, rotation='vertical')
    #plt.yticks(tick_marks, index_dic, rotation=0)
    
    plt.margins(0.5)
    
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.15)

    '''
    fmt = '.2f' 
    thresh = dist_matrix.max() / 2.
    for i, j in itertools.product(range(dist_matrix.shape[0]), range(dist_matrix.shape[1])):
        plt.text(j, i, format(dist_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if dist_matrix[i, j] > thresh else "black")
        
    '''

    #plt.tight_layout()
    
    plt.ylabel('Registered Faces')
    plt.xlabel('Registered Faces')
    
    ax.grid(True,color='red')
    ticklines = ax.get_xticklines() + ax.get_yticklines()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    ticklabels = ax.get_xticklabels() + ax.get_yticklabels()
    
    for line in ticklines:
        line.set_linewidth(6)
        
    
    for line in gridlines:
        line.set_linestyle('-.')

    for label in ticklabels:
        label.set_color('r')
        label.set_fontsize('medium')



def box_plot_distances(reg_faces_encodings, face_name_index_map):
    '''
        to show intra, within   class distance variation
    
    '''
    for name in face_name_index_map:
        img_indx_list = face_name_index_map[name]
        intra_dists = reg_faces_encodings[img_indx_list, img_indx_list]
        print(intra_dists.shape)
    

if __name__ == '__main__':
    
    hdf5_dataset_fullName = os.path.join(REG_FACES_PATH, HDF5_FACE_DATASET_NAME)
    
    hdf5_file = h5py.File(hdf5_dataset_fullName, 'r')
    
    aligned_faces = hdf5_file["face_imgs_cv"]
    
    index_dic =  hdf5_file["index_dic"][:]
    
    reg_faces_encodings = hdf5_file["faces_encoding"][:]
    
    # get unique person names
    registerd_person_names = list(set(index_dic)) 
    
    
    
    print(registerd_person_names)
    
    print(registerd_person_names[0])
    
    face_name_index_map ={}
    
    for i, person_name in enumerate(index_dic):
        #print(i, person_name, len(person_name))
        #if any(person_name in s for s in registerd_person_names):
        if(len(person_name) != 0):
            face_name_index_map[person_name] = []
            
    for i, person_name in enumerate(index_dic):
        #print(i, person_name, len(person_name))
        #if any(person_name in s for s in registerd_person_names):
        if(len(person_name) != 0):
            face_name_index_map.get(person_name).append(i)
    
    # Compute distance matrix
    dist_mat = get_distances_matrix(reg_faces_encodings, index_dic)
    np.set_printoptions(precision=2)
    # Plot distance matrix
    
    #plot_dist_matrix(dist_mat, index_dic, face_name_index_map)
    #plt.show()
    plot_dist_seaborn(dist_mat, index_dic,face_name_index_map)
    
        
    #box_plot_distances(reg_faces_encodings, face_name_index_map)
    
    for name in face_name_index_map:
        img_indx_list = face_name_index_map[name]
        
        print("name",name)
        print("img_indx_list", img_indx_list)
        img_list=[]
        for index in img_indx_list:
            img_list.append(aligned_faces[index])
            
        #montage_shape = (round(math.sqrt(len(img_list))), round(math.sqrt(len(img_list))))
        montage_shape = (10,1)
            
        faces_montages = build_montages(img_list, image_shape=(192,192),
                                                montage_shape=montage_shape)
        
        for num, faces_montage in enumerate(faces_montages):
            cv2.imshow(name + "  " + str(num), faces_montage)
            cv2.imwrite("./images/" + name + "-" + str(num) + ".jpg" , faces_montage)
    
        if cv2.waitKey(0) == ord('q') :
            break
        
        
        
  
    cv2.destroyAllWindows()
    '''
       import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

name_list = ('Omar', 'Serguey', 'Max', 'Zhou', 'Abidin')
value_list = np.random.randint(0, 99, size = len(name_list))
pos_list = np.arange(len(name_list))

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.FixedLocator((pos_list)))
ax.xaxis.set_major_formatter(ticker.FixedFormatter((name_list)))

plt.bar(pos_list, value_list, color = '.75', align = 'center')
plt.show()
    ''' 
    
    