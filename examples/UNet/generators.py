__author__ = 'Fabian Isensee'
import numpy as np

def batch_generator(data, target, BATCH_SIZE):
    '''
    just a simple batch iterator, no cropping, no rotation, no anything
    '''
    np.random.seed()
    idx = np.arange(data.shape[0])
    while True:
        ids = np.random.choice(idx, BATCH_SIZE)
        yield np.array(data[ids]), np.array(target[ids])

def random_crop_generator(generator, crop_size=(128, 128)):
    '''
    yields a random crop of size crop_size
    '''
    np.random.seed()
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
