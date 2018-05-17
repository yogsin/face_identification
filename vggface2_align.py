#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import math
#from python_wrapper import *
import os
import sys

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
def walk_through_the_folder_for_crop(src_path, label_file, dest_path):
    n_p = 0
    i = 0
    img_count = 0
    f = open(label_file, 'r')
    for labels in f.readlines():
        if i == 0:
            i += 1
            continue
        labels = labels.strip()
        label = labels.split(',')
        point = []
        point.extend(float(x) for x in label[1:10:2])
        point.extend(float(y) for y in label[2:11:2])
        im_identity, im_name = label[0][1:-1].split('/')
        img_dir = src_path + im_identity +'/' + im_name + '.jpg'
        if not os.path.exists(img_dir):
            continue
        if not os.path.exists(dest_path + '/' + im_identity +'/'):
            os.mkdir(dest_path + '/' + im_identity +'/')
        dest_file_path = dest_path + '/' + im_identity +'/' + im_name + '.jpg'
        #print type(im)
        im = Image.open(img_dir)
        face_slign(im, point, dest_file_path)
        img_count += 1
        #if img_count > 5:
            #break
        i += 1
        #if i > 0:
            #break
    print 'total %d identities %d images' % (i, img_count)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    src_path = sys.argv[1]
    label_file = sys.argv[2]
    dest_path = sys.argv[3]
    if not src_path.endswith('/'):
        src_path += '/'
    if not os.path.exists(dest_path + '/'):
        os.mkdir(dest_path + '/')
    #main()
    walk_through_the_folder_for_crop(src_path, label_file, dest_path)
