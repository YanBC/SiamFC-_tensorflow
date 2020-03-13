import os
import pickle
from random import shuffle, choice
import cv2 as cv
import sys
sys.path.append('.')

from utils.bbox import xywh2xyxy


#########################
# unit classes
#########################

class Data:
    '''
    basic data unit

    properties
    --------
    imageName: string
        relative image path

    coors: list of int
        [left, top, right, bottom]
    '''
    def __init__(self, imageName, coors):
        self.imageName = imageName
        self.coors = coors


class Sequence_Data(list):
    '''
    this class holds data for a whole video clip
    '''
    def __init__(self, sequenceDir):
        super().__init__()
        self.dir = sequenceDir



#########################
# dataset classes
#########################


class Sequence_Dataset_Base:
    '''
    base class for all datasets

    properties
    --------
    dir: stirng
        path to dataset directory

    positive_interval: int
        maximum interval between images for them to be 
        considered as a positive pair

    sequence_datas: list of Sequence_Data
        the list of video clips

    sequence_ids: list of int
        ids of video clips, should have the same length 
        as sequence_datas

    storage: immutable string
        where to store the saved sequence_datas
    '''
    def __init__(self, path, positive_interval=100):
        self.dir = path
        self.positive_interval = positive_interval
        self.sequence_datas = None
        self.sequence_ids = None

    @property # to make storage immutable
    def storage(self):
        s_path = './sequenced_datasets'
        if not os.path.isdir(s_path):
            os.mkdir(s_path)
        return s_path

    def show_image(self, randomly=False):
        if self.sequence_datas is None:
            print('You have to load squence first')
            return False

        if randomly:
            shuffle(self.sequence_ids)

        windowName = 'show'
        cv.namedWindow(windowName, cv.WINDOW_NORMAL)
        for i in self.sequence_ids:
            sequence = self.sequence_datas[i]

            while True:
                data = choice(sequence)
                imageName = os.path.join(self.dir, sequence.dir, data.imageName)
                coors = data.coors

                image = cv.imread(imageName)
                left, top, right, bottom = coors

                cv.rectangle(image, (left, top), (right, bottom), (0,255,0), 3)
                cv.imshow(windowName, image)
                ch = cv.waitKey()
                if chr(ch) == 'q' or chr(ch) == 'n':
                    break

            if chr(ch) == 'q':
                break

    def shuffle(self, rng):
        if self.sequence_datas is None:
            print('You have to load squence first')
            return False
        rng.shuffle(self.sequence_ids)
        return True


    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # rewrite these functions for each dataset
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def load_squence(self):
        pass

    def load_sequence_from_file(self, srcPath):
        pass

    def save_sequence_to_file(self, desName):
        pass





class GOT10k_Dataset(Sequence_Dataset_Base):

    def load_sequence(self):
        print('''GOT10k_Dataset: loading data...''')
        listfile = 'list.txt'
        ignorefile = './data/unfixed_got10k_list.txt'
        with open(os.path.join(self.dir, listfile)) as f:
            videoDirs = sorted([x.strip() for x in f.readlines()])

        with open(ignorefile) as f:
            videoDirsIgnore = sorted([x.strip() for x in f.readlines()])

        for ignore in videoDirsIgnore:
            videoDirs.remove(ignore)

        sequence_list = []
        for videoDir in videoDirs:
            sequence_dir = os.path.join(self.dir, videoDir)
            imageFiles = sorted([x for x in os.listdir(sequence_dir) if '.jpg' in x])
            with open(os.path.join(sequence_dir, 'groundtruth.txt')) as f:
                annos = [x.strip() for x in f.readlines()]
            assert len(annos) == len(imageFiles)

            sequence = Sequence_Data(videoDir)

            for i in range(len(annos)):
                imageFile = imageFiles[i]
                coors = annos[i].split(',')
                left, top, width, height = [int(float(x)) for x in coors]
                right = left + width - 1
                bottom = top + height - 1
                coors = [left, top, right, bottom]

                sequence.append(Data(imageFile, coors))

            sequence_list.append(sequence)

        self.sequence_datas = sequence_list
        self.sequence_ids = [x for x in range(len(self.sequence_datas))]
        print('''finish loading''')
        return True

    def load_sequence_from_file(self, srcPath):
        print('''GOT10k_Dataset: loading data...''')
        with open(srcPath, 'br') as f:
            self.sequence_datas = pickle.load(f)
        self.sequence_ids = [x for x in range(len(self.sequence_datas))]
        print('''finish loading''')
        return True

    def save_sequence_to_file(self, desName='got10k.pkl'):
        print('''GOT10k_Dataset: saving data...''')
        if self.sequence_datas is None:
            print('You have to load squence first')
            return False
        desPath = os.path.join(self.storage, desName)
        with open(desPath, 'bw') as f:
            pickle.dump(self.sequence_datas, f)
        print('''finish saving ''')
        return True




# test GOT10k
if __name__ == '__main__':
  got_path = './datasets/GOT10k/train_data/'
  dataName = 'got10k_filered.pkl'

  got_dataset = GOT10k_Dataset(got_path, positive_interval=100)
  got_dataset.load_sequence()
  got_dataset.save_sequence_to_file(desName=dataName)
  del got_dataset

  got_dataset = GOT10k_Dataset(got_path, positive_interval=100)
  got_dataset.load_sequence_from_file(os.path.join(got_dataset.storage, dataName))
  got_dataset.show_image()




#########################
# formater classes
#########################
class Siamfcpp_Formater:
    def __init__(self):
        pass

    def __call__(self):
        pass





#########################
# sampler classes
#########################
class Siamfcpp_Sampler:
    def __init__(self):
        pass

    def sample_one(self, rng):
        pass