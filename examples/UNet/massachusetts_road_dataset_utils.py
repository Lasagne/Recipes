__author__ = 'Fabian Isensee'
import numpy as np
import os
import sys
import fnmatch
import matplotlib.pyplot as plt
sys.path.append("../../modelzoo/")
from time import sleep
from generators import *

def prep_folders():
    if not os.path.isdir("data"):
        os.mkdir("data")

    if not os.path.isdir("data/validation"):
        os.mkdir("data/validation")
    if not os.path.isdir("data/training"):
        os.mkdir("data/training")
    if not os.path.isdir("data/test"):
        os.mkdir("data/test")

    if not os.path.isdir("data/validation/sat_img"):
        os.mkdir("data/validation/sat_img")
    if not os.path.isdir("data/validation/map"):
        os.mkdir("data/validation/map")
    if not os.path.isdir("data/training/sat_img"):
        os.mkdir("data/training/sat_img")
    if not os.path.isdir("data/training/map"):
        os.mkdir("data/training/map")
    if not os.path.isdir("data/test/sat_img"):
        os.mkdir("data/test/sat_img")
    if not os.path.isdir("data/test/map"):
        os.mkdir("data/test/map")

def prep_urls():
    valid_data_url = valid_target_url = np.loadtxt("mass_roads_validation.txt", dtype=str)
    valid_data_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/"
    valid_target_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/"

    train_data_url = train_target_url  = np.loadtxt("mass_roads_train.txt", dtype=str)
    train_data_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/"
    train_target_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/"

    test_data_url = test_target_url  = np.loadtxt("mass_roads_test.txt", dtype=str)
    test_data_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/"
    test_target_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/"

    f = open("mass_roads_train_data_download.sh", 'w')
    g = open("mass_roads_train_target_download.sh", 'w')
    for img_name in train_data_url:
        f.write("wget -O data/training/sat_img/%sf "%img_name + train_data_str + img_name + "f" + "\n")
        g.write("wget -O data/training/map/%s "%img_name + train_target_str + img_name + "\n")
    f.close()
    g.close()

    f = open("mass_roads_validation_data_download.sh", 'w')
    g = open("mass_roads_validation_target_download.sh", 'w')
    for img_name in valid_data_url:
        f.write("wget -O data/validation/sat_img/%s "%img_name + valid_data_str + img_name + "\n")
        g.write("wget -O data/validation/map/%s "%img_name[:-1] + valid_target_str + img_name[:-1] + "\n")
    f.close()
    g.close()

    f = open("mass_roads_test_data_download.sh", 'w')
    g = open("mass_roads_test_target_download.sh", 'w')
    for img_name in test_data_url:
        f.write("wget -O data/test/sat_img/%s "%img_name + test_data_str + img_name + "\n")
        g.write("wget -O data/test/map/%s "%img_name[:-1] + test_target_str + img_name[:-1] + "\n")
    f.close()
    g.close()

def download_dataset():
    os.system("sh mass_roads_train_data_download.sh &")
    os.system("sh mass_roads_train_target_download.sh &")
    os.system("sh mass_roads_validation_data_download.sh &")
    os.system("sh mass_roads_validation_target_download.sh &")
    os.system("sh mass_roads_test_data_download.sh &")
    os.system("sh mass_roads_test_target_download.sh &")

def load_data(folder):
    images_sat = [img for img in os.listdir(os.path.join(folder, "sat_img")) if fnmatch.fnmatch(img, "*.tif*")]
    images_map = [img for img in os.listdir(os.path.join(folder, "map")) if fnmatch.fnmatch(img, "*.tif*")]
    assert(len(images_sat) == len(images_map))
    images_sat.sort()
    images_map.sort()
    # images are 1500 by 1500 pixels each
    data = np.zeros((len(images_sat), 3, 1500, 1500), dtype=np.uint8)
    target = np.zeros((len(images_sat), 1, 1500, 1500), dtype=np.uint8)
    ctr = 0
    for sat_im, map_im in zip(images_sat, images_map):
        data[ctr] = plt.imread(os.path.join(folder, "sat_img", sat_im)).transpose((2, 0, 1))
        # target has values 0 and 255. make that 0 and 1
        target[ctr, 0] = plt.imread(os.path.join(folder, "map", map_im))/255
        ctr += 1
    return data, target

def prepare_dataset():
    prep_folders()
    prep_urls()
    download_dataset()
    # the dataset is now downloaded in the background. Once every few seconds we check if the download is done. We do
    # this by checking whether tha last training image exists
    while not os.path.isfile("data/training/map/99238675_15.tiff") and not os.path.isfile("data/training/sat_img/99238675_15.tiff"):
        print "download seems to be running..."
        sleep(5)
    print "download done..."
    try:
        data_train, target_train = load_data("data/training")
        data_valid, target_valid = load_data("data/validation")
        data_test, target_test = load_data("data/test")
        # loading np arrays is much faster than loading the images one by one every time
        np.save("train_data.npy", data_train)
        np.save("train_target.npy", target_train)
        np.save("valid_data.npy", data_valid)
        np.save("valid_target.npy", target_valid)
        np.save("test_data.npy", data_test)
        np.save("test_target.npy", target_test)
    except:
        print "something went wrong, maybe the download?"