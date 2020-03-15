import numpy as np
from multiprocessing import Queue, Process, Event, Lock


def create_worker(sampler, queue, exit_event):
    rng = np.random.RandomState()

    while not exit_event.is_set():
        data = sampler.sample_one(rng)
        try:
            queue.put(data, timeout=2)
        except:
            pass



class DataLoader:
    def __init__(self, sampler, num_worker=4, buffer_size=6):
        self.queue = Queue(buffer_size)
        self.exit = Event()
        self.workers = [Process(target=create_worker, args=(sampler, self.queue, self.exit,)) for i in range(num_worker)]

        for worker in self.workers:
            worker.start()

    def load_one(self):
        return self.queue.get()

    def shutdown(self):
        # self.exit.set()
        # for worker in self.workers:
        #     worker.join()
        for worker in self.workers:
            worker.terminate()
        return True



if __name__ == '__main__':
    import os
    import cv2 as cv
    import sys
    sys.path.append('.')
    from data.classification_datasets import Data, Imagenet2012, Alexnet_Formater, Alexnet_Sampler

    imagenet_dir = './datasets/imagenet'
    dataName = 'imagenet2012.pkl'
    dataset = Imagenet2012(imagenet_dir)
    dataset.load_data_from_file(os.path.join(dataset.storage, dataName))

    input_size = 224
    input_channel = 3
    channel_mean = dataset.channel_mean
    num_cls = 1000
    formater = Alexnet_Formater(input_size, channel_mean, num_cls)

    batchsize = 1
    sampler = Alexnet_Sampler(dataset, formater, batchsize)

    datagen = DataLoader(sampler, num_worker=2, buffer_size=16)

    ch = ord(' ')
    while True:
        train_data = datagen.load_one()
        image = train_data['X'][0]
        label = train_data['Y'][0]
        if chr(ch) == 'n':
            if label.argmax() != target_label:
                continue
        image += dataset.channel_mean
        image = image.astype(np.uint8)
        print(image.shape)

        windowName = 'show'
        cv.namedWindow(windowName, cv.WINDOW_NORMAL)
        cv.imshow(windowName, image)
        ch = cv.waitKey()
        if chr(ch) == 'q':
            break
        elif chr(ch) == 'n':
            target_label = label.argmax()

    tmp = datagen.shutdown()
