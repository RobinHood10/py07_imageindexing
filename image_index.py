#!/usr/bin/python

import argparse
import cv2
import sys
import os
import re
import cPickle as pickle
import fnmatch

########################
# module: image_index.py
# Silvia Smith
# A01396094
########################

ap = argparse.ArgumentParser()
ap.add_argument('-imgdir', '--imgdir', required = True, help = 'image directory')
ap.add_argument('-bgr', '--bgr', required = True, help = 'bgr index file to pickle')
ap.add_argument('-hsv', '--hsv', required = True, help = 'hsv index file to pickle')
ap.add_argument('-gsl', '--gsl', required = True, help = 'gsl index file to pickle')
args = vars(ap.parse_args())

def generate_file_names(fnpat, rootdir):
  #copied from hw06
  for dirpath, dirname, filename in os.walk(rootdir):
    for file_name in fnmatch.filter(filename, fnpat):
      yield os.path.join(dirpath, file_name)

## three index dictionaries
HSV_INDEX = {}
BGR_INDEX = {}
GSL_INDEX = {}

def index_img(imgp):
    try:
        img = cv2.imread(imgp)
        index_bgr(imgp, img)
        index_hsv(imgp, img)
        index_gsl(imgp, img)
        del img
    except Exception, e:
        print(str(e))

#TODO
# img = images in directory
# imgp = hash table (dictionary)

# compute the bgr vector for img saved in path imgp and index it in BGR_INDEX under 
# imgp.
def index_bgr(imgp, img):
  (height, width, num_channels) = img.shape
  B,G,R = cv2.split(img)
  indexList = []
  for b, g, r in zip(B,G,R):
    bmu = sum(b)*1.0/len(b)    
    gmu = sum(g)*1.0/len(g)
    rmu = sum(g)*1.0/len(r)
    indexList.append(bmu, gmu, rmu)
  BGR_INDEX[imgp,indexList]

# compute the hsv vector for img saved in path imgp and
# index it in HSV_INDEX under imgp.
def index_hsv(imgp, img):
  hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  H,S,V = cv2.split(hsv_image)
  indexList = []
  for h,s,v in zip(H,S,V):
    hmu = sum(h)*1.0/len(h)
    smu = sum(s)*1.0/len(s)
    vmu = sum(v)*1.0/len(v)
    indexList.append(hmu, smu, vmu)
  HSV_INDEX[imgp, indexList]

# compute the grayscale vector for img saved in path imgp and
# index it in GSL_INDEX under imgp.
def index_gsl(imgp, img):
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  GR = cv2.split(gray_image)
  indexList = []
  for gr in GR:
    grmu = sum(gr)*1.0/len(gr)
    indexList.append(grmu)
  GSL_INDEX[imgp, indexList]

# index image directory imgdir
def index_img_dir(imgdir):
 print(imgdir)
 for imgp in generate_file_names(r'.+\.(jpg|png|JPG)', imgdir):
    print('indexing ' + imgp)
    index_img(imgp)
    print(imgp + ' indexed')

# index and pickle
if __name__ == '__main__':
  index_img_dir(args['imgdir'])
  with open(args['bgr'], 'wb') as bgrfile:
    pickle.dump(BGR_INDEX, bgrfile)
  with open(args['hsv'], 'wb') as hsvfile:
    pickle.dump(HSV_INDEX, hsvfile)
  with open(args['gsl'], 'wb') as gslfile:
    pickle.dump(GSL_INDEX, gslfile)
  print('indexing finished')

