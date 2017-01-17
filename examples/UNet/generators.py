__author__ = 'Fabian Isensee'
import numpy as np
import lasagne

def batch_generator(data, target, BATCH_SIZE, shuffle=False):
    if shuffle:
        while True:
            ids = np.random.choice(len(data), BATCH_SIZE)
            yield data[ids], target[ids]
    else:
        for idx in range(0, len(data), BATCH_SIZE):
            ids = slice(idx, idx + BATCH_SIZE)
            yield data[ids], target[ids]

def batch_generator_old(data, target, BATCH_SIZE, shuffle=False):
    '''
    just a simple batch iterator, no cropping, no rotation, no anything
    '''
    np.random.seed()
    idx = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(idx)
    idx_2 = np.array(idx)
    # if BATCH_SIZE is larger than len(data) we need to artificially enlarge the idx array (loop around)
    while BATCH_SIZE > len(idx):
        idx_2 = np.concatenate((idx_2, idx))
    del(idx)
    while True:
        ctr = 0
        yield np.array(data[idx_2[ctr:ctr+BATCH_SIZE]]), np.array(target[idx_2[ctr:ctr+BATCH_SIZE]])
        ctr += BATCH_SIZE
        if ctr >= data.shape[0]:
            ctr -= data.shape[0]

def center_crop_generator(generator, output_size):
    '''
    yields center crop of size output_size (may be 1d or 2d) from data and seg
    '''
    '''if type(output_size) not in (tuple, list):
        center_crop = [output_size, output_size]
    elif len(output_size) == 2:
        center_crop = list(output_size)
    else:
        raise ValueError("invalid output_size")'''
    center_crop = lasagne.utils.as_tuple(output_size, 2, int)
    for data, seg in generator:
        center = np.array(data.shape[2:])/2
        yield data[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.)], seg[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.)]


def random_crop_generator(generator, crop_size=(128, 128)):
    '''
    yields a random crop of size crop_size
    '''
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size, crop_size]
    elif len(crop_size) == 2:
        crop_size = list(crop_size)
    else:
        raise ValueError("invalid crop_size")
    for data, seg in generator:
        lb_x = np.random.randint(0, data.shape[2]-crop_size[0])
        lb_y = np.random.randint(0, data.shape[3]-crop_size[1])
        data = data[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1]]
        seg = seg[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1]]
        yield data, seg

def threaded_generator(generator, num_cached=10):
    # this code is written by jan Schluter
    # copied from https://github.com/benanne/Lasagne/issues/12
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()
