import os
import pickle
import xmltodict
from random import shuffle
import cv2 as cv
import numpy as np
import multiprocessing.dummy as md


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

    label: int
        an interger defining the class
        of the object
    '''
    def __init__(self, imagePath, coors, label):
        self.imagePath = imagePath
        self.coors = coors
        self.label = label


#########################
# dataset classes
#########################
class Classification_Dataset_Base:
    def __init__(self, path):
        self.dir = path
        self.imageDir = os.path.join(self.dir, 'images')
        self.annoDir = os.path.join(self.dir, 'annotations')
        self.num_cls = None
        self.data = None
        self.data_ids = None

    @property
    def size(self):
        if self.data_ids is None:
            print('You have to load data first')
            return -1
        else:
            return len(self.data_ids)
    
    @property # to make storage immutable
    def storage(self):
        s_path = './classification_datasets'
        if not os.path.isdir(s_path):
            os.mkdir(s_path)
        return s_path

    def show_image(self, randomly=False):
        if self.data is None:
            print('You have to load data first')
            return False

        if randomly:
            shuffle(self.data_ids)

        windowName = 'show'
        cv.namedWindow(windowName, cv.WINDOW_NORMAL)
        ch = ord(' ')
        for index in self.data_ids:
            data = self.data[self.data_ids[index]]
            if chr(ch) == 'n':
                if data.label != target_label:
                    continue
            imageLocation = os.path.join(self.imageDir, data.imagePath)
            coors = data.coors
            print(f'Class: #{data.label}\nImagePath: {imageLocation}')

            image = cv.imread(imageLocation)
            left, top, right, bottom = coors

            cv.rectangle(image, (left, top), (right, bottom), (0,255,0), 3)
            cv.imshow(windowName, image)
            ch = cv.waitKey()
            if chr(ch) == 'q':
                break
            elif chr(ch) == 'n':
                target_label = data.label

    def shuffle(self, rng):
        if self.data_ids is None:
            print('You have to load data first')
            return False
        rng.shuffle(self.sequence_ids)
        return True


    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # rewrite these functions for each dataset
    ##@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def load_data(self):
        pass

    def load_data_from_file(self, srcPath):
        pass

    def save_data_to_file(self, desName):
        pass


class Imagenet2012(Classification_Dataset_Base):

    def _read_anno(self, args):
        dirname, xmlname = args
        imagename = xmlname.split('.')[0] + '.JPEG'
        xml_path = os.path.join(self.annoDir, dirname, xmlname)
        with open(xml_path) as f:
            annotation = xmltodict.parse(f.read())

        imagepath = os.path.join(dirname, imagename)
        label = dirname
        obj = annotation['annotation']['object']
        if isinstance(obj, list):
            obj = obj[0]
        left = int(obj['bndbox']['xmin'])
        top = int(obj['bndbox']['ymin'])
        right = int(obj['bndbox']['xmax'])
        bottom = int(obj['bndbox']['ymax'])
        return (imagepath, [left, top, right, bottom], label)

    def load_data(self):
        print('''Imagenet2012: loading data...''')
        args = []
        for dirname in os.listdir(self.annoDir):
            for xmlname in os.listdir(os.path.join(self.annoDir, dirname)):
                args.append((dirname, xmlname))
        with md.Pool(processes=8) as p:
            data_list = p.map(self._read_anno, args)

        # with open('./temp/annos.pkl', 'bw') as f:
        #     pickle.dump(data_list, f)
        # with open('./temp/annos.pkl', 'br') as f:
        #     data_list = pickle.load(f)

        label_list = sorted(list(os.listdir(self.imageDir)))
        self.label_dict = dict()
        for i in range(len(label_list)):
            self.label_dict[label_list[i]] = i

        self.data = []
        for d in data_list:
            imagePath = d[0]
            coors = d[1]
            label = self.label_dict[d[2]]
            self.data.append(Data(imagePath=imagePath, coors=coors, label=label))
        self.data_ids = [x for x in range(len(self.data))]

        print('''finish loading''')
        return True

    def load_data_from_file(self, srcPath):
        print('''Imagenet2012: loading data from file...''')
        with open(srcPath, 'br') as f:
            self.data, self.label_dict = pickle.load(f)
        self.data_ids = [x for x in range(len(self.data))]
        print('''finish loading''')
        return True

    def save_data_to_file(self, desName='imagenet2012.pkl'):
        print('''Imagenet2012: saving data...''')
        if self.data is None:
            print('You have to load data first')
            return False
        desPath = os.path.join(self.storage, desName)
        with open(desPath, 'bw') as f:
            pickle.dump((self.data, self.label_dict), f)
        print('''finish saving''')
        return True

    def cal_mean(self):
        if not self.channel_mean is None:
            return self.channel_mean
        else:
            if self.data is None:
                print('You have to load data first')
                return np.array([-1, -1, -1])

            data_mean_sum = np.array([0,0,0])
            for data in self.data:
                imagePath = os.path.join(self.imageDir, data.imagePath)
                image = cv.imread(imagePath)
                data_mean_sum += np.mean(image, axis=(0,1))

            channel_mean = data_mean_sum / len(self.data)
            self.channel_mean = channel_mean
            return channel_mean



# # test Imagenet2012
# if __name__ == '__main__':
#     imagenet_dir = './datasets/imagenet'
#     dataName = 'imagenet2012.pkl'
#     # imagenet_dataset = Imagenet2012(imagenet_dir)
#     # imagenet_dataset.load_data()
#     # imagenet_dataset.save_data_to_file(desName=dataName)
#     # del imagenet_dataset

#     imagenet_dataset = Imagenet2012(imagenet_dir)
#     imagenet_dataset.load_data_from_file(os.path.join(imagenet_dataset.storage, dataName))
#     imagenet_dataset.show_image(randomly=True)





#########################
# formater classes
#########################
class Alexnet_Formater:
    '''
    Data formater for AlexNet

    properties
    --------
    target_size: int
        size of the target image, target
        image will be resize to be of 
        shape (target_size, target_size, 3)

    channel_mean: list of float
        mean of each RGB channel over the
        entire train set, in the form
        of [r_mean, g_mean, b_mean]

    num_cls: int
        total number of classes
    '''
    def __init__(self, size, channel_mean, num_cls):
        self.target_size = size
        self.channel_mean = channel_mean
        self.num_cls = num_cls

    def __call__(image, coors, label):
        left, top, right, bottom = coors
        object_image = image[top:bottom, left:right, :]
        scaled_image = self._rescale(object_image)
        crop_image = self._crop_centre(scaled_image)
        mean_image = self._subtract_mean(crop_image)
        final_image = np.expand_dims(mean_image, axis=0)

        final_label = self._one_hot(label)

        return {'x': final_image, 'y': final_label}

    def _rescale(img):
        h, w, _ = img.shape
        if h < w:
            shorter_side = h
            scale = self.target_size / shorter_side
            target_h = self.target_size
            target_w = scale * w
        else:
            shorter_side = w
            scale = self.target_size / shorter_side
            target_h = scale * h
            target_w = self.target_size

        scaled_img = cv.resize(img, dsize=(target_w, target_h))
        return scaled_img

    def _crop_centre(img, c_x, c_y):
        h, w, _ = img.shape

        half_size = self.target_size // 2
        if self.target_size % 2 == 0:
            crop_img = img[c_y-half_size:c_y+half_size, c_x-half_size:c_x+half_size, :]
        else:
            crop_img = img[c_y-half_size:c_y+half_size+1, c_x-half_size:c_x+half_size+1, :]

        return crop_img

    def _subtract_mean(img):
        h, w, c = img.shape
        assert c == 3
        return img - self.channel_mean

    def _one_hot(c):
        base = np.zeros(shape=(1, self.num_cls))
        base[c] = 1
        return base




#########################
# sampler classes
#########################
class Alexnet_Sampler:
    def __init__(self, dataset, formater, batchsize):
        self.dataset = dataset
        self.formater = formater
        self.batchsize = batchsize

    def sample_one(self, rng):
        imageDir = self.dataset.imageDir
        all_data_ids = self.dataset.data_ids

        indice = rng.choice(all_data_ids, size=self.batchsize, replace=False)
        X = []
        Y = []
        for index in indice:
            data = self.dataset.data[all_data_ids[index]]
            imagePath = data.imagePath
            image = cv.imread(imagePath)
            coors = data.coors
            label = data.label

            formated = self.formater(image, coors, label)
            X.append(formated['x'])
            Y.append(formated['y'])

        return {'X': X, 'Y':Y}

