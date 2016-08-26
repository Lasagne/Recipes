__author__ = 'Fabian Isensee'
import os
import sys
import fnmatch
import matplotlib.pyplot as plt
sys.path.append("../../modelzoo/")
from generators import *
from multiprocessing.dummy import Pool
from urllib import urlretrieve

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

    all_tasks = []

    # save url along with the filename for each file
    for img_name in train_data_url:
        all_tasks.append(tuple([train_data_str + img_name + "f", "data/training/sat_img/%sf"%img_name]))
        all_tasks.append(tuple([train_target_str + img_name, "data/training/map/%s"%img_name]))

    for img_name in valid_data_url:
        all_tasks.append(tuple([valid_data_str + img_name, "data/validation/sat_img/%s"%img_name]))
        all_tasks.append(tuple([valid_target_str + img_name[:-1], "data/validation/map/%s"%img_name[:-1]]))

    for img_name in test_data_url:
        all_tasks.append(tuple([test_data_str + img_name, "data/test/sat_img/%s"%img_name]))
        all_tasks.append(tuple([test_target_str + img_name[:-1], "data/test/map/%s"%img_name[:-1]]))

    return all_tasks

def download_dataset(all_tasks, num_workers=4):
    def urlretrieve_star(args):
        return urlretrieve(*args)

    pool = Pool(num_workers)
    pool.map(urlretrieve_star, all_tasks)
    pool.close()
    pool.join()


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
    all_tasks = prep_urls()
    download_dataset(all_tasks)

    print "download done..."
    try:
        data_train, target_train = load_data("data/training")
        data_valid, target_valid = load_data("data/validation")
        data_test, target_test = load_data("data/test")
        # loading np arrays is much faster than loading the images one by one every time
        np.savez_compressed("road_segm_dataset.npz",
                            data_train=data_train, target_train=target_train,
                            data_valid=data_valid, target_valid=target_valid,
                            data_test=data_test, target_test=target_test)
    except:
        print "something went wrong, maybe the download?"

if __name__ == "__main__":
    prepare_dataset()