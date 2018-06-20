#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

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
def walk_through_the_folder_for_crop(src_path, dst_path):
    i = 0
    #print 'the folders contain more than 100 image are:'
    for people_folder in os.listdir(src_path):
        i += 1
        people_path_src = src_path + people_folder + '/'
        people_path_dst = dst_path + people_folder + '/'
        if len(os.listdir(people_path_src)) != len(os.listdir(people_path_dst)):
            print people_folder + " is different"
        sys.stdout.write('\rtotal: %d folder, %d folder done ' % (len(os.listdir(src_path)), i) )
        sys.stdout.flush()
    print "done"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(1)
    src_path = sys.argv[1]
    dst_path = sys.argv[2]
    if not src_path.endswith('/'):
        src_path += '/'
    if not dst_path.endswith('/'):
        dst_path += '/'

    #main()
    walk_through_the_folder_for_crop(src_path, dst_path)
