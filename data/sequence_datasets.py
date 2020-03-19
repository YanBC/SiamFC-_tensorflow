import os
import pickle
from random import shuffle, choice
import cv2 as cv
import sys
sys.path.append('.')

from utils.bbox import xywh2xyxy
from utils.filter_box import filter_unreasonable_training_boxes
from utils.crop_track_pair import crop_track_pair
from utils.make_densebox_target import make_densebox_target


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
            print('You have to load sequence first')
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
            print('You have to load sequence first')
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
            print('You have to load sequence first')
            return False
        desPath = os.path.join(self.storage, desName)
        with open(desPath, 'bw') as f:
            pickle.dump(self.sequence_datas, f)
        print('''finish saving ''')
        return True




# # test GOT10k
# if __name__ == '__main__':
#   got_path = './datasets/GOT10k/train_data/'
#   dataName = 'got10k_filered.pkl'

#   # got_dataset = GOT10k_Dataset(got_path, positive_interval=100)
#   # got_dataset.load_sequence()
#   # got_dataset.save_sequence_to_file(desName=dataName)
#   # del got_dataset

#   got_dataset = GOT10k_Dataset(got_path, positive_interval=100)
#   got_dataset.load_sequence_from_file(os.path.join(got_dataset.storage, dataName))
#   got_dataset.show_image()




# #########################
# # formater classes
# #########################
# class Siamfcpp_Formater:
#     def __init__(self):
#         pass

#     def __call__(self):
#         pass





#########################
# sampler classes
#########################
class Siamfcpp_Sampler:
    def __init__(self, dataset, batchsize, neg_ratio=0.1):
        self.dataset = dataset
        self.batchsize = batchsize
        self.neg_ratio = neg_ratio

    def sample_one(self, rng):
        data = []
        for i in range(self.batchsize):
            is_negative_pair = rng.rand() < self.neg_ratio

            if is_negative_pair:
                data1 = self._sample_frame(rng)
                data2 = self._sample_frame(rng)
            else:
                data1, data2 = self._sample_pair(rng)

            sampled_data = dict(data_z=data1, data_x=data2, is_negative_pair=is_negative_pair)
            croped_data = self._crop_data(sampled_data)
            target_data = self._make_target(croped_data)
            data.append(target_data)

        im_z = np.stack([x['im_z'] for x in data])
        im_x = np.stack([x['im_x'] for x in data])
        bbox_z = np.stack([x['bbox_z'] for x in data])
        bbox_x = np.stack([x['bbox_x'] for x in data])
        cls_gt = np.stack([x['cls_gt'] for x in data])
        ctr_gt = np.stack([x['ctr_gt'] for x in data])
        box_gt = np.stack([x['box_gt'] for x in data])

        final_data = dict(
            im_z=im_z,
            im_x=im_x,
            bbox_z=bbox_z,
            bbox_x=bbox_x,
            cls_gt=cls_gt,
            ctr_gt=ctr_gt,
            box_gt=box_gt
        )
        return final_data

    def _sample_frame(self, rng):
        x_id = rng.choice(self.dataset.sequence_ids)
        sequence = self.dataset.sequence_datas[x_id]

        data = rng.choice(sequence)
        image = self._load_image(os.path.join(sequence.dir, data.imageName))
        coors = data.coors
        return dict(image=image, bbox=coors)

    def _sample_pair(self, rng):
        x_id = rng.choice(self.dataset.sequence_ids)
        sequence = self.dataset.sequence_datas[x_id]

        max_diff = self.dataset.positive_interval
        L = len(sequence)
        idx1 = rng.choice(L)
        idx2_choices = list(range(idx1-max_diff, L)) + list(range(L+1, idx1+max_diff+1))
        idx2_choices = list(set(idx2_choices).intersection(set(range(L))))
        idx2 = rng.choice(idx2_choices)
        
        data1 = sequence[idx1]
        image1 = self._load_image(os.path.join(sequence.dir, data1.imageName))
        coors1 = data1.coors
        data2 = sequence[idx2]
        image2 = self._load_image(os.path.join(sequence.dir, data2.imageName))
        coors2 = data2.coors

        return dict(image=image1, bbox=coors1), dict(image=image2, bbox=coors2)

    def _load_image(self, imagePath):
        image = cv.imread(imagePath)
        if image is None:
            print(f'{imagePath} not exits')
            return None
        else:
            return image

    def _crop_data(self, data_pair):
        crop_config = dict(context_amount=1, max_scale=0.3, max_shift=0.4, max_scale_temp=0, max_shift_temp=0, z_size=127, x_size=303,)

        data_z = data_pair['data_z']
        data_x = data_pair['data_x']

        im_temp, bbox_temp = data_z['image'], data_z['bbox']
        im_curr, bbox_curr = data_x['image'], data_x['bbox']
        im_z, bbox_z, im_x, bbox_x = crop_track_pair(im_temp, bbox_temp, im_curr, bbox_curr, crop_config)

        ret_pair = dict()
        ret_pair['data_z'] = dict(image=im_z, bbox=bbox_z)
        ret_pair['data_x'] = dict(image=im_x, bbox=bbox_x)
        ret_pair['is_negative_pair'] = data_pair['is_negative_pair']
        return ret_pair

    def _make_target(self, data_pair):
        target_config = dict(z_size=127, x_size=303, score_size=17, score_offset=87, total_stride=8, num_conv3x3=3)
        target_config['score_size'] = (target_config['x_size'] - target_config['z_size']) // target_config['total_stride'] + 1 - target_config['num_conv3x3'] * 2
        target_config['score_offset'] = (target_config['x_size'] - 1 - (target_config['score_size'] - 1) * target_config['total_stride']) // 2

        data_z = data_pair['data_z']
        im_z, bbox_z = data_z['image'], data_z['bbox']
        data_x = data_pair['data_x']
        im_x, bbox_x = data_x['image'], data_x['bbox']
        is_negative_pair = data_pair['is_negative_pair']

        cls_label, ctr_label, box_label = make_densebox_target(bbox_x.reshape(1, 4), target_config)
        if is_negative_pair:
            cls_label[cls_label == 0] = -1
            cls_label[cls_label == 1] = 0

        target_data = dict(
            im_z=im_z,
            im_x=im_x,
            bbox_z=bbox_z,
            bbox_x=bbox_x,
            cls_gt=cls_label,
            ctr_gt=ctr_label,
            box_gt=box_label,
            is_negative_pair=int(is_negative_pair),
        )

        return target_data


