#!/usr/bin/python

import argparse
import cv2
import sys
import os
import re
import cPickle as pickle

########################
# module: image_retrieval.py
# Silvia Smith
# A01396094
########################

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--imgpath', required = True, help = 'image path')
ap.add_argument('-bgr', '--bgr', required = True, help = 'bgr index file to unpickle')
ap.add_argument('-hsv', '--hsv', required = True, help = 'hsv index file to unpickle')
ap.add_argument('-gsl', '--gsl', required = True, help = 'gsl index file to unpickle')
args = vars(ap.parse_args())

def mean(v):
  return sum(v)/(len(v)*1.0)

# compute the bgr similarity between
# two bgr index vectors
def bgr_img_sim(img_index_vec1, img_index_vec2):
  # your code
  pass
  
# compute the hsv similarity between
# two hsv index vectors
def hsv_img_sim(img_index_vec1, img_index_vec2):
  # your code
  pass

# compute the hsv similarity between
# two gsl index vectors
def gsl_img_sim(img_index1, img_index2):
  # your code
  pass

# index the input image
def index_img(imgp):
    try:
        img = cv2.imread(imgp)
        if img is None:
          print('cannot read ' + imgp)
          return
        rslt = (index_bgr(img), index_hsv(img), index_gsl(img))
        del img
        return rslt
    except Exception, e:
        print(str(e))

# this is very similar to index_bgr in image_index.py except
# you do not have to save the index in BGR_INDEX. This index
# is used to match the indices in the unpickeld BGR_INDEX.
def index_bgr(img):
    # your code
    pass

# this is very similar to index_hsv in image_index.py except
# you do not have to save the index in HSV_INDEX. This index
# is used to match the indices in the unpickeld HSV_INDEX.
def index_hsv(img):
    # your code
    pass

# this is very similar to index_gs. in image_index.py except
# you do not have to save the index in GSL_INDEX. This index
# is used to match the indices in the unpickeld GSL_INDEX.
def index_gsl(img):
    # your code
    pass

# we will unpickle into these global vars below.
BGR_INDEX = None
HSV_INDEX = None
GSL_INDEX = None

# compute the similarities between the bgr
# index vector and all the vectors in the unpickled
# bgr index bgr_index and return the top one.
def compute_bgr_sim(bgr, bgr_index, topn=1):
  # your code 
  pass

# compute the similarities between the hsv
# index vector and all the vectors in the unpickled
# hsv index hsv_index and return the top one.
def compute_hsv_sim(hsv, hsv_index, topn=1):
  # your code
  pass

# compute the similarities between the gsl
# index vector and all the vectors in the unpickled
# gsl index gls_index and return the top one.
def compute_gsl_sim(gsl, gsl_index, topn=1):
  # your code
  pass

# unpickle, match, and display
if __name__ == '__main__':
  with open(args['bgr'], 'rb') as bgrfile:
    BGR_INDEX = pickle.load(bgrfile)
  with open(args['hsv'], 'rb') as hsvfile:
    HSV_INDEX = pickle.load(hsvfile)
  with open(args['gsl'], 'rb') as gslfile:
    GSL_INDEX = pickle.load(gslfile)

  bgr, hsv, gsl = index_img(args['imgpath'])
  bgr_matches = compute_bgr_sim(bgr, BGR_INDEX)
  hsv_matches = compute_hsv_sim(hsv, HSV_INDEX)
  gsl_matches = compute_gsl_sim(gsl, GSL_INDEX)

  print bgr_matches
  print hsv_matches
  print gsl_matches

  orig = cv2.imread(args['imgpath'])
  bgr = cv2.imread(bgr_matches[0][0])
  hsv = cv2.imread(hsv_matches[0][0])
  gsl = cv2.imread(hsv_matches[0][0])
  cv2.imshow('Input', orig)
  cv2.imshow('BGR', bgr)
  cv2.imshow('HSV', hsv)
  cv2.imshow('GSL', gsl)
  cv2.waitKey()
  del orig
  del bgr
  del hsv
  del gsl
  cv2.destroyAllWindows()
    

