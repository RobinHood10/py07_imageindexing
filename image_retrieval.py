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
# blue_sim = dot product(b) / sqrt(sum(b**2)*sum(pb**2))
  for b1,g1,r1,b2,g2,r2 in zip(img_index_vec1, img_index_vec2):
    bluedot = sum(b1*b2)
    greendot = sum(g1*g2)
    reddot = sum(r1*r2)

    b1SQsum = sum(b1**2); b2DQsum = sum(b2**2)
    g1SQsum = sum(g1**2); g2SQsum = sum(g2**2)
    r1SQsum = sum(r1**2); r2SQsum = sum(r2**2)   

  blue_sim = bluedot/sqrt(b1SQsum*b2SQsum)
  green_sim = greendot/sqrt(g1SQsum*g2SQsum)
  red_sim = reddot/sqrt(r1SQsum*r2sqsum)
  rgb_sim_final = (blue_sim + green_sim + red_sim)/3.0
  return rgb_sim_final

# compute the hsv similarity between
# two hsv index vectors
def hsv_img_sim(img_index_vec1, img_index_vec2):
  for h1,s1,v1,h2,s2,v2 in zip(img_index_vec1, img_index_vec2):
    huedot = sum(h1*h2)
    satdot = sum(s1*s2)
    valdot = sum(v1*v2)

    h1SQsum = sum(h1**2); h2DQsum = sum(h2**2)
    s1SQsum = sum(s1**2); s2SQsum = sum(s2**2)
    v1SQsum = sum(v1**2); v2SQsum = sum(v2**2)   

  hue_sim = huedot/sqrt(h1SQsum*h2SQsum)
  sat_sim = satdot/sqrt(s1SQsum*s2SQsum)
  val_sim = valdot/sqrt(v1SQsum*v2sqsum)
  hsv_sim_final = (hue_sim + sat_sim + val_sim)/3.0
  return hsv_sim_final

# compute the hsv similarity between
# two gsl index vectors
def gsl_img_sim(img_index1, img_index2):
  for gr1, gr2 in zip(img_index1, img_index2):
    greydot = sum(gr1*gr2)
    gr1SQsum = sum(gr1**2)
    gr2SQsum = sum(gr2**2)
  gsl_sim = greydot/sqrt(gr1SQsum*gr2SQsum)
  return gsl_sim

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
  B,G,R = cv2.split(img)
  indexList = []
  for b,g,r in zip(B,G,R):
    bmu = sum(b)*1.0/len(b)
    gmu = sum(g)*1.0/len(g)
    rmu = sum(r)*1.0/len(r)
    indexList.append(bmu, gmu, rmu)
  return indexList

# this is very similar to index_hsv in image_index.py except
# you do not have to save the index in HSV_INDEX. This index
# is used to match the indices in the unpickeld HSV_INDEX.
def index_hsv(img):
  hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  H,S,V = cv2.split(hsv_image)
  indexList = []
  for h,s,v in zip(H,S,V):
    hmu = sum(h)*1.0/len(h)
    smu = sum(s)*1.0/len(s)
    vmu = sum(v)*1.0/len(v)
    indexList.append(hmu, smu, vmu)
  return indexList

# this is very similar to index_gs. in image_index.py except
# you do not have to save the index in GSL_INDEX. This index
# is used to match the indices in the unpickeld GSL_INDEX.
def index_gsl(img):
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  GR = cv2.split(gray_image)
  indexList = []
  for gr in GR:
    grmu = sum(gr)*1.0/len(gr)
    indexList.append(grmu)
  return indexList

# we will unpickle into these global vars below.
BGR_INDEX = None
HSV_INDEX = None
GSL_INDEX = None

# compute the similarities between the bgr
# index vector and all the vectors in the unpickled
# bgr index bgr_index and return the top one.
def compute_bgr_sim(bgr, bgr_index, topn=1):
  #bgr index vector = 1 vector
  # bgr_index = dictionary of tons of vectors
  sim_index = []
  for vector in bgr_index:
    sim_index.append(vector, bgr_img_sim(bgr,vector))
  sim_index = sorted(sim_index, key=lambda x: x[1])
  sim_deliverable = []
  for x in xrange(topn):
    sim_deliverable.append(sim_index[1])
  return sim_deliverable

# compute the similarities between the hsv
# index vector and all the vectors in the unpickled
# hsv index hsv_index and return the top one.
def compute_hsv_sim(hsv, hsv_index, topn=1):
  sim_index = []
  for vector in hsv_index:
    sim_index.append(vector, hsv_img_sim(hsv,vector))
  sim_index = sorted(sim_index, key=lambda x: x[1])
  sim_deliverable = []
  for x in xrange(topn):
    sim_deliverable.append(sim_index[1])
  return sim_deliverable

# compute the similarities between the gsl
# index vector and all the vectors in the unpickled
# gsl index gls_index and return the top one.
def compute_gsl_sim(gsl, gsl_index, topn=1):
  sim_index = []
  for vector in gsl_index:
    sim_index.append(vector, gsl_img_sim(gsl,vector))
  sim_index = sorted(sim_index, key=lambda x: x[1])
  sim_deliverable = []
  for x in xrange(topn):
    sim_deliverable.append(sim_index[1])
  return sim_deliverable

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
