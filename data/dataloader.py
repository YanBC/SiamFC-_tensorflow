import numpy as np
from multiprocessing import Queue, Process, Event, Lock


def create_worker(index, sampler, queue, lock, exit_event, neg_ratio=0.1):
    rng = np.random.RandomState(index)

    while not exit_event.is_set():
        if rng.rand() < neg_ratio:
            isNegative = True
        else:
            isNegative = False
        data = sampler.sample_one(rng, isNegative)
        if lock.acquire(timeout=2):
            try:
                queue.put(data, timeout=2)
            except:
                pass
            finally:
                lock.release()



class DataLoader:
    def __init__(self, sampler, num_worker=4, neg_ratio=0.1, buffer_size=6):
        self.queue = Queue(buffer_size)
        self.queue_lock = Lock()
        self.exit = Event()
        self.workers = [Process(target=create_worker, args=(i, sampler, self.queue, self.queue_lock, self.exit, neg_ratio,)) for i in range(num_worker)]

        for worker in self.workers:
            worker.start()

    def load_one(self):
        return self.queue.get()

    def shutdown(self):
        self.exit.set()
        return True



if __name__ == '__main__':
    class Sample:
        def __init__(self):
            self.numbers = [x for x in range(10)]

        def sample_one(self, rng, isNegative):
            return rng.choice(self.numbers)

    s = Sample()
    d = DataLoader(s)

    for i in range(10):
        print(d.load_one())

    d.shutdown()