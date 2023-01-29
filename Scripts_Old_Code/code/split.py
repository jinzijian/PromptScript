
import xml.dom.minidom
from transformers import *
import logging
import os


def get_all_file(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                L.append(os.path.join(root, file))
    return L
l = get_all_file("/home/zijian/Scripts_new/all_data_temp_train")
print(l)

#print(len("/home/zijian/Scripts_new/all_data_temp"))
#train_dir = "/home/zijian/Scripts_new/all_data_temp_train"
#test_dir =   "/home/zijian/Scripts_new/all_data_temp_test"
#dev_dir =   "/home/zijian/Scripts_new/all_data_temp_dev"
#
#def save_file(filename,new_dir):
#    f1 = open(filename)
#    f2 = open(new_dir + filename[38:], 'w')
#    print(new_dir + filename[38:])
#    f2.write(f1.read())
#    f1.close()
##ßprint(l)
#import random
#random.seed(42)
#random.shuffle(l)
#
##print(len(l))
##
##print(l[:5])
#train_size = (int)(0.5 * len(l))
#dev_size = (int)(0.2 * len(l))
#test_size = len(l) - train_size - dev_size
#
#print(train_size)
#print(dev_size)
#print(test_size)
#
#
#for filename in l[:train_size]:
#    save_file(filename, train_dir)
#
#
#for filename in l[train_size : train_size + dev_size]:
#    save_file(filename, dev_dir)
#
#
#for filename in l[train_size + dev_size:]:
#    save_file(filename, test_dir)
    
