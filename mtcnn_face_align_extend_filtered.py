#!/usr/bin/env python
# -*- coding: utf-8 -*-

import _init_paths
from PIL import Image
import caffe
import cv2
import numpy as np
import math
#from python_wrapper import *
import os
import sys

def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print("reshape of reg")
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    #print("bb", boundingbox)
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!
    #print('#################')
    #print('boxes', boxes)
    #print('w,h', w, h)
    
    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    numbox = boxes.shape[0]

    #print('tmph', tmph)
    #print('tmpw', tmpw)

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0:1][:,0]
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]
    ey = boxes[:,3:4][:,0]
   
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)
    
    #print("dy"  ,dy )
    #print("dx"  ,dx )
    #print("y "  ,y )
    #print("x "  ,x )
    #print("edy" ,edy)
    #print("edx" ,edx)
    #print("ey"  ,ey )
    #print("ex"  ,ex )


    #print('boxes', boxes)
    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]



def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]
    h = bboxA[:,3] - bboxA[:,1]
    l = np.maximum(w,h).T
    
    #print('bboxA', bboxA)
    #print('w', w)
    #print('h', h)
    #print('l', l)
    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA


def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort()) # read s using I
    
    pick = [];
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]
    return pick


def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0,:,:].T
    dy1 = reg[1,:,:].T
    dx2 = reg[2,:,:].T
    dy2 = reg[3,:,:].T
    (x, y) = np.where(map >= t)

    yy = y
    xx = x
    
    '''
    if y.shape[0] == 1: # only one point exceed threshold
        y = y.T
        x = x.T
        score = map[x,y].T
        dx1 = dx1.T
        dy1 = dy1.T
        dx2 = dx2.T
        dy2 = dy2.T
        # a little stange, when there is only one bb created by PNet
        
        #print("1: x,y", x,y)
        a = (x*map.shape[1]) + (y+1)
        x = a/map.shape[0]
        y = a%map.shape[0] - 1
        #print("2: x,y", x,y)
    else:
        score = map[x,y]
    '''
    #print("dx1.shape", dx1.shape)
    #print('map.shape', map.shape)
   

    score = map[x,y]
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([yy, xx]).T

    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
    score = np.array([score])

    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)

    #print('(x,y)',x,y)
    #print('score', score)
    #print('reg', reg)

    return boundingbox_out.T



def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
    return im

def drawPoints(im, points):
    x1 = points[:,0]
    x2 = points[:,1]
    x3 = points[:,2]
    x4 = points[:,3]
    x5 = points[:,4]
    y1 = points[:,5]
    y2 = points[:,6]
    y3 = points[:,7]
    y4 = points[:,8]
    y5 = points[:,9]
    print x1,y1
    for i in range(x1.shape[0]):
        cv2.circle(im,(x1,y1),3,(0,0,255),-1)
        cv2.circle(im,(x2,y2),3,(0,0,255),-1)
        cv2.circle(im,(x3,y3),3,(0,0,255),-1)
        cv2.circle(im,(x4,y4),3,(0,0,255),-1)
        cv2.circle(im,(x5,y5),3,(0,0,255),-1)
    return im

from time import time
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    print(fmt % (time()-_tstart_stack.pop()))


def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):
    
    img2 = img.copy()

    factor_count = 0
    total_boxes = np.zeros((0,9), np.float)
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0/minsize
    minl = minl*m
    

    #total_boxes = np.load('total_boxes.npy')
    #total_boxes = np.load('total_boxes_242.npy')
    #total_boxes = np.load('total_boxes_101.npy')

    
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    
    # first stage
    for scale in scales:
        hs = int(np.ceil(h*scale))
        ws = int(np.ceil(w*scale))

        if fastresize:
            im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
        else: 
            im_data = cv2.resize(img, (ws,hs)) # default is bilinear
            im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]
        #im_data = imResample(img, hs, ws); print("scale:", scale)


        im_data = np.swapaxes(im_data, 0, 2)
        im_data = np.array([im_data], dtype = np.float)
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()
    
        boxes = generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
        if boxes.shape[0] != 0:
            #print(boxes[4:9])
            #print('im_data', im_data[0:5, 0:5, 0], '\n')
            #print('prob1', out['prob1'][0,0,0:3,0:3])

            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0 :
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
         
    #np.save('total_boxes_101.npy', total_boxes)

    #####
    # 1 #
    #####
    print("[1]:",total_boxes.shape[0])
    #print(total_boxes)
    #return total_boxes, [] 


    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        print("[2]:",total_boxes.shape[0])
        
        # revise and convert to square
        regh = total_boxes[:,3] - total_boxes[:,1]
        regw = total_boxes[:,2] - total_boxes[:,0]
        t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        t4 = total_boxes[:,3] + total_boxes[:,8]*regh
        t5 = total_boxes[:,4]
        total_boxes = np.array([t1,t2,t3,t4,t5]).T
        #print("[3]:",total_boxes.shape[0])
        #print(regh)
        #print(regw)
        #print('t1',t1)
        #print(total_boxes)

        total_boxes = rerec(total_boxes) # convert box to square
        print("[4]:",total_boxes.shape[0])
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        print("[4.5]:",total_boxes.shape[0])
        #print(total_boxes)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

    #print(total_boxes.shape)
    #print(total_boxes)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage

        #print('tmph', tmph)
        #print('tmpw', tmpw)
        #print("y,ey,x,ex", y, ey, x, ex, )
        #print("edy", edy)

        #tempimg = np.load('tempimg.npy')

        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
          
            #print("dx[k], edx[k]:", dx[k], edx[k])
            #print("dy[k], edy[k]:", dy[k], edy[k])
            #print("img.shape", img[y[k]:ey[k]+1, x[k]:ex[k]+1].shape)
            #print("tmp.shape", tmp[dy[k]:edy[k]+1, dx[k]:edx[k]+1].shape)

            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
            #print("y,ey,x,ex", y[k], ey[k], x[k], ex[k])
            #print("tmp", tmp.shape)
            
            tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))
            #tempimg[k,:,:,:] = imResample(tmp, 24, 24)
            #print('tempimg', tempimg[k,:,:,:].shape)
            #print(tempimg[k,0:5,0:5,0] )
            #print(tempimg[k,0:5,0:5,1] )
            #print(tempimg[k,0:5,0:5,2] )
            #print(k)
    
        #print(tempimg.shape)
        #print(tempimg[0,0,0,:])
        tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python

        #np.save('tempimg.npy', tempimg)

        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        #print(tempimg[0,:,0,0])
        
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()

        #print(out['conv5-2'].shape)
        #print(out['prob1'].shape)

        score = out['prob1'][:,1]
        #print('score', score)
        pass_t = np.where(score>threshold[1])[0]
        #print('pass_t', pass_t)
        
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)
        print("[5]:",total_boxes.shape[0])
        #print(total_boxes)

        #print("1.5:",total_boxes.shape)
        
        mv = out['conv5-2'][pass_t, :].T
        #print("mv", mv)
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            #print('pick', pick)
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                print("[6]:",total_boxes.shape[0])
                total_boxes = bbreg(total_boxes, mv[:, pick])
                print("[7]:",total_boxes.shape[0])
                total_boxes = rerec(total_boxes)
                print("[8]:",total_boxes.shape[0])
            
        #####
        # 2 #
        #####
        print("2:",total_boxes.shape)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)
           
            #print('tmpw', tmpw)
            #print('tmph', tmph)
            #print('y ', y)
            #print('ey', ey)
            #print('x ', x)
            #print('ex', ex)
        

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                
            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()
            
            score = out['prob1'][:,1]
            points = out['conv6-3']
            pass_t = np.where(score>threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
            print("[9]:",total_boxes.shape[0])
            
            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:,3] - total_boxes[:,1] + 1
            h = total_boxes[:,2] - total_boxes[:,0] + 1

            points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
            points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                print("[10]:",total_boxes.shape[0])
                pick = nms(total_boxes, 0.7, 'Min')
                
                #print(pick)
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    print("[11]:",total_boxes.shape[0])
                    points = points[pick, :]

    #####
    # 3 #
    #####
    print("3:",total_boxes.shape)

    return total_boxes, points




    
def initFaceDetector():
    minsize = 20
    caffe_model_path = "/home/duino/iactive/mtcnn/model"
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
    return (minsize, PNet, RNet, ONet, threshold, factor)

def haveFace(img, facedetector):
    minsize = facedetector[0]
    PNet = facedetector[1]
    RNet = facedetector[2]
    ONet = facedetector[3]
    threshold = facedetector[4]
    factor = facedetector[5]
    
    if max(img.shape[0], img.shape[1]) < minsize:
        return False, []

    img_matlab = img.copy()
    tmp = img_matlab[:,:,2].copy()
    img_matlab[:,:,2] = img_matlab[:,:,0]
    img_matlab[:,:,0] = tmp
    
    #tic()
    boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    #toc()
    containFace = (True, False)[boundingboxes.shape[0]==0]
    return containFace, boundingboxes

def face_slign(im, points, dest_file_path):
    w,h = im.size
    angle = math.atan((points[6] - points[5]) / (points[1] - points[0])) * 180 / math.pi    #角度制，用于旋转图片
    angle_rad = - math.atan((points[6] - points[5]) / (points[1] - points[0]))    #弧度制，用于坐标计算
    #print 'angle: %f' %angle
    if angle == 0.0:
        im_rotate = im
    else:
        im_rotate = im.rotate(angle)

    w_rotate,h_rotate = im_rotate.size
    #旋转之前左眼右眼鼻尖嘴角坐标，因为是相对图像中心点旋转，所以需要预先作平移变换
    le_x = points[0] - w/2.0
    le_y = points[5] - h/2.0
    re_x = points[1] - w/2.0
    re_y = points[6] - h/2.0
    n_x = points[2] - w/2.0
    n_y = points[7] - h/2.0
    lm_x = points[3] - w/2.0
    lm_y = points[8] - h/2.0
    rm_x = points[4] - w/2.0
    rm_y = points[9] - h/2.0

    #旋转之后的坐标，由于坐标经过平移，此处需平移回去
    rotate_le_x = le_x * math.cos(angle_rad) - le_y * math.sin(angle_rad) + w_rotate/2.0
    rotate_le_y = le_x * math.sin(angle_rad) + le_y * math.cos(angle_rad) + h_rotate/2.0
    rotate_re_x = re_x * math.cos(angle_rad) - re_y * math.sin(angle_rad) + w_rotate/2.0
    rotate_re_y = re_x * math.sin(angle_rad) + re_y * math.cos(angle_rad) + h_rotate/2.0
    rotate_n_x = n_x * math.cos(angle_rad) - n_y * math.sin(angle_rad) + w_rotate/2.0
    rotate_n_y = n_x * math.sin(angle_rad) + n_y * math.cos(angle_rad) + h_rotate/2.0
    rotate_lm_x = lm_x * math.cos(angle_rad) - lm_y * math.sin(angle_rad) + w_rotate/2.0
    rotate_lm_y = lm_x * math.sin(angle_rad) + lm_y * math.cos(angle_rad) + h_rotate/2.0
    rotate_rm_x = rm_x * math.cos(angle_rad) - rm_y * math.sin(angle_rad) + w_rotate/2.0
    rotate_rm_y = rm_x * math.sin(angle_rad) + rm_y * math.cos(angle_rad) + h_rotate/2.0

    # #眼睛中心点之间的水平距离
    # e_d = (rotate_re_x - rotate_le_x)
    # #眼睛中心到鼻尖的垂直距离
    # e_n_d = rotate_n_y - (rotate_le_y + rotate_re_y) / 2
    # #鼻子中心点与嘴巴中心点垂直距离
    # n_m_d = (rotate_lm_y + rotate_rm_y) / 2 - rotate_n_y
    #眼睛中心点与嘴巴中心点垂直距离
    e_m_d = (rotate_lm_y + rotate_rm_y) / 2 - (rotate_le_y + rotate_re_y) / 2

    # #根据距离对图像进行裁剪，缩放
    # scale_w = 35.0 / 58    #相对左右眼中心点两边扩充35/58（双眼中心点水平距离）
    # scale_u = 33.0 / 38    #相对左右眼中心点中心往上扩充33/38（眼睛中心点与鼻子中心点垂直距离）
    # scale_d = 2.0    #相对左右嘴角中心往下扩充2倍（鼻子中心点与嘴巴中心点垂直距离）
    #根据期望的眼睛中心点与嘴巴中心点垂直距离进行缩放，首先计算缩放比例
    scale_d = 38.0/e_m_d
    if w_rotate * scale_d > 0 or h_rotate * scale_d > 0:
        resized_im = im_rotate.resize((int(w_rotate * scale_d), int(h_rotate * scale_d)), Image.ANTIALIAS)

        resized_w,resized_h = resized_im.size
        #缩放后双眼中心点与嘴巴中心点坐标
        resized_e_x = (rotate_le_x + rotate_re_x) / 2 * scale_d
        resized_e_y = (rotate_le_y + rotate_re_y) / 2 * scale_d
        resized_m_x = (rotate_lm_x + rotate_rm_x) / 2 * scale_d
        resized_m_y = (rotate_lm_y + rotate_rm_y) / 2 * scale_d
        #扩展后的坐标，添加了防止越界判断
        l_x = np.max((0,resized_e_x - 48))
        r_x = np.min((resized_w,resized_e_x + 48))
        u_y = np.max((0,resized_e_y - 36))
        d_y = np.min((resized_h,resized_m_y + 36))
        #print l_x,r_x,u_y,d_y
        #print dest_file_path
        #im_rotate = alignment(im, points)
        #裁剪图像
        crop_im = resized_im.crop((l_x, u_y, r_x, d_y))
        #缩放到指定大小，保存最终对齐的图像
        final_im = crop_im.resize((96,112), Image.ANTIALIAS)
        final_im.save(dest_file_path)
        #resized_im = crop_im.resize((47,55), Image.ANTIALIAS)
        #resized_im = crop_im.resize((96,112), Image.ANTIALIAS)
        #resized_im.save(dest_file_path)

def filter_face(points, w, h):
    x = abs(points[:,0:5] - w/2)
    y = abs(points[:,5:10] - h/2)
    abs_dis_min_index = (x + y).sum(axis=1).argmin(axis=0)
    return abs_dis_min_index
#webface只有两级目录
'''
总目录
|-- identity 1
|    |-- image 1
|    |--...
|    |-- image n
|-- ...
|-- identity n
|    |-- image 1
|    |--...
|    |-- image n
'''
def walk_through_the_folder_for_crop(src_path, dest_path):
    minsize = 20
    caffe_model_path = "./model"
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    caffe.set_mode_gpu()
    caffe.set_device(3)
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)

    n_p = 0
    i = 0
    img_count = 0
    #print 'the folders contain more than 100 image are:'
    for people_folder in os.listdir(src_path):
        #n_p = n_p + 1
        #if n_p < 7042:
            #continue
        people_path = src_path + people_folder + '/'
        for img_name in os.listdir(people_path):
            img_dir = people_path + img_name
            img = cv2.imread(img_dir)
            [h, w, c] = img.shape
            img_matlab = img.copy()
            tmp = img_matlab[:,:,2].copy()
            img_matlab[:,:,2] = img_matlab[:,:,0]
            img_matlab[:,:,0] = tmp

            # check rgb position
            #tic()
            boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
            print points
            #point = points[0]
            #未检测到人脸，则跳出循环处理下一张图片
            if not boundingboxes.shape[0] > 0:
                continue 
            if boundingboxes.shape[0] == 1:
                point = points[0]
            if boundingboxes.shape[0] > 1:
                abs_dis_min_index = filter_face(points, w, h)
                point = points[abs_dis_min_index]
            print point
            if not os.path.exists(dest_path + '/' + people_folder +'/'):
                os.mkdir(dest_path + '/' + people_folder +'/')
            dest_file_path = dest_path + '/' + people_folder +'/' + img_name
            #print type(im)
            im = Image.open(img_dir)
            face_slign(im, point, dest_file_path)
            img_count += 1
            #if img_count > 5:
                #break
        i += 1
        #if i > 0:
            #break
        sys.stdout.write('\rtotal: %d identities, %d identities done' % (len(os.listdir(src_path)), i) )
        sys.stdout.flush()
    print 'total %d identities %d images' % (i, img_count)

def main():
    #imglistfile = "./file.txt"
    imglistfile = "imglist.txt"
    #imglistfile = "/home/duino/iactive/mtcnn/all.txt"
    #imglistfile = "./imglist.txt"
    #imglistfile = "/home/duino/iactive/mtcnn/file_n.txt"
    #imglistfile = "/home/duino/iactive/mtcnn/file.txt"
    minsize = 20

    caffe_model_path = "./model"

    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)


    #error = []
    f = open(imglistfile, 'r')
    for imgpath in f.readlines():
        imgpath = imgpath.split('\n')[0]
        print("######\n", imgpath)
        img = cv2.imread(imgpath)
        img_matlab = img.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp

        # check rgb position
        #tic()
        boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
        #toc()

        ## copy img to positive folder
        #if boundingboxes.shape[0] > 0 :
        #    import shutil
        #    shutil.copy(imgpath, '/home/duino/Videos/3/disdata/positive/'+os.path.split(imgpath)[1] )
        #else:
        #    import shutil
        #    shutil.copy(imgpath, '/home/duino/Videos/3/disdata/negetive/'+os.path.split(imgpath)[1] )

        # useless org source use wrong values from boundingboxes,case uselsee rect is drawed 
#        for i in range(len(boundingboxes)):
#            cv2.rectangle(img, (int(boundingboxes[i][0]), int(boundingboxes[i][1])), (int(boundingboxes[i][2]), int(boundingboxes[i][3])), (0,255,0), 1)    

        img = drawBoxes(img, boundingboxes)
        img = drawPoints(img, points)
        cv2.imwrite("result.jpg", img)
        cv2.imshow('img', img)
        ch = cv2.waitKey(0) & 0xFF
        if ch == 27:
            break


        #if boundingboxes.shape[0] > 0:
        #    error.append[imgpath]
    #print(error)
    f.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    src_path = sys.argv[1]
    dest_path = sys.argv[2]
    if not src_path.endswith('/'):
        src_path += '/'
    if not os.path.exists(dest_path + '/'):
        os.mkdir(dest_path + '/')
    #main()
    walk_through_the_folder_for_crop(src_path, dest_path)
