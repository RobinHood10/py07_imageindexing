#!/usr/bin/python

import argparse
import cv2
import sys
import os
import re
import cPickle as pickle

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

# compute the bgr vector for img saved in path imgp and
# index it in BGR_INDEX under imgp.
def index_bgr(imgp, img):
    # your code
    pass

# compute the hsv vector for img saved in path imgp and
# index it in HSV_INDEX under imgp.
def index_hsv(imgp, img):
    # your code
    pass

# compute the gsl vector for img saved in path imgp and
# index it in GSL_INDEX under imgp.
def index_gsl(imgp, img):
  # your code
  pass

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
    

