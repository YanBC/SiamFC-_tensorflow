import os
import pickle
import xmltodict
from random import shuffle
import cv2 as cv
import numpy as np
import multiprocessing.dummy as md
import multiprocessing as mp


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
    def __init__(self, imagePath, coors, label, size):
        self.imagePath = imagePath
        self.coors = coors
        self.label = label
        self.size = size


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
        width = int(annotation['annotation']['size']['width'])
        height = int(annotation['annotation']['size']['height'])
        obj = annotation['annotation']['object']
        if isinstance(obj, list):
            obj = obj[0]
        left = int(obj['bndbox']['xmin'])
        top = int(obj['bndbox']['ymin'])
        right = int(obj['bndbox']['xmax'])
        bottom = int(obj['bndbox']['ymax'])

        image = cv.imread(os.path.join(self.imageDir, imagepath))
        h, w, c = image.shape
        if h != height or w != width or c != 3:
            return None
        else:
            return (imagepath, [left, top, right, bottom], label, [width, height])

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
            if d is None:
                continue
            imagePath = d[0]
            coors = d[1]
            label = self.label_dict[d[2]]
            size = d[3]
            self.data.append(Data(imagePath=imagePath, coors=coors, label=label, size=size))
        self.data_ids = [x for x in range(len(self.data))]

        print('''finish loading''')
        return True

    def load_data_from_file(self, srcPath):
        print('''Imagenet2012: loading data from file...''')
        with open(srcPath, 'br') as f:
            self.data, self.label_dict, self._channel_mean = pickle.load(f)
        self.data_ids = [x for x in range(len(self.data))]
        print('''finish loading''')
        return True

    def save_data_to_file(self, desName):
        print('''Imagenet2012: saving data...''')
        if self.data is None:
            print('You have to load data first')
            return False
        desPath = os.path.join(self.storage, desName)
        with open(desPath, 'bw') as f:
            channel_mean = self.channel_mean
            pickle.dump((self.data, self.label_dict, channel_mean), f)
        print('''finish saving''')
        return True

    def _cal_mean(self, imagePath):
        image = cv.imread(imagePath)
        if image is None:
            print(f'None: {imagePath}')
            return np.array([0.,0.,0.])
        return np.mean(image, axis=(0,1))

    @property
    def channel_mean(self):
        if hasattr(self, '_channel_mean'):
            return self._channel_mean
        else:
            if self.data is None:
                print('You have to load data first')
                return np.array([-1., -1., -1.])

            imagePaths = [os.path.join(self.imageDir, data.imagePath) for data in self.data]
            with mp.Pool(processes=16) as p:
                data_means = p.map(self._cal_mean, imagePaths)
            self._channel_mean = np.stack(data_means).mean(axis=0)
            return self._channel_mean


class Imagenet2012_Val(Imagenet2012):

    def __init__(self, path, label_dict=None, channel_mean=None):
        super().__init__(path)
        assert os.path.isdir(self.annoDir)
        assert os.path.isdir(self.imageDir)
        self.label_dict = label_dict
        self._channel_mean = channel_mean

    def _read_anno(self, args):
        imagename, xmlname = args

        imagepath = os.path.join(self.imageDir, imagename)
        xmlpath = os.path.join(self.annoDir, xmlname)
        image = cv.imread(imagepath)
        with open(xmlpath) as f:
            annotation = xmltodict.parse(f.read())

        width = int(annotation['annotation']['size']['width'])
        height = int(annotation['annotation']['size']['height'])
        obj = annotation['annotation']['object']
        if isinstance(obj, list):
            obj = obj[0]
        left = int(obj['bndbox']['xmin'])
        top = int(obj['bndbox']['ymin'])
        right = int(obj['bndbox']['xmax'])
        bottom = int(obj['bndbox']['ymax'])
        label = obj['name']

        h, w, c = image.shape
        if (h != height) or (w != width) or (c != 3) or (label not in self.label_dict.keys()):
            return None
        else:
            return (imagename, [left, top, right, bottom], label, [width, height])

    def load_data(self):
        print('Imagenet2012: loading data...')
        if self.label_dict is None or self._channel_mean is None:
            print('In order to load data, you have to initialize Imagenet2012_Val with a valid label_dict and a valid channel_mean. Exiting...')
            return False

        imageFiles = sorted(os.listdir(self.imageDir))
        annoFiles = sorted(os.listdir(self.annoDir))
        args = [x for x in zip(imageFiles, annoFiles)]

        with md.Pool(processes=8) as p:
            data_list = p.map(self._read_anno, args)

        self.data = []
        for d in data_list:
            if d is None:
                continue
            imagePath = d[0]
            coors = d[1]
            label = self.label_dict[d[2]]
            size = d[3]
            self.data.append(Data(imagePath=imagePath, coors=coors, label=label, size=size))
        self.data_ids = [x for x in range(len(self.data))]

        print('''finish loading''')
        return True

    @property
    def channel_mean(self):
        return self._channel_mean


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

    def __call__(self, image, coors, label):
        left, top, right, bottom = coors
        object_image = image[top:bottom+1, left:right+1, :]
        scaled_image = self._rescale(object_image)
        scaled_image_h, scaled_image_w, _ = scaled_image.shape
        crop_image = self._crop_centre(scaled_image, scaled_image_w//2, scaled_image_h//2)
        mean_image = self._subtract_mean(crop_image)
        final_image = np.expand_dims(mean_image, axis=0)

        final_label = self._one_hot(label)

        return {'x': final_image, 'y': final_label}

    def _rescale(self, img):
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

        scaled_img = cv.resize(img, dsize=(int(np.ceil(target_w)), int(np.ceil(target_h))))
        return scaled_img

    def _crop_centre(self, img, c_x, c_y):
        h, w, _ = img.shape

        half_size = self.target_size // 2
        if self.target_size % 2 == 0:
            crop_img = img[c_y-half_size:c_y+half_size, c_x-half_size:c_x+half_size, :]
        else:
            crop_img = img[c_y-half_size:c_y+half_size+1, c_x-half_size:c_x+half_size+1, :]

        return crop_img

    def _subtract_mean(self, img):
        h, w, c = img.shape
        assert c == 3
        return img - self.channel_mean

    def _one_hot(self, c):
        base = np.zeros(shape=(1, self.num_cls))
        base[0, c] = 1
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
            data = self.dataset.data[index]
            imagePath = os.path.join(self.dataset.imageDir, data.imagePath)
            image = cv.imread(imagePath)
            coors = data.coors
            label = data.label
            # width, height = data.size
            # image_height, image_width, _ = image.shape
            # if width != image_width or height != image_height:
            #     image.transpose([1,0,2])

            formated = self.formater(image, coors, label)
            X.append(formated['x'])
            Y.append(formated['y'])

        X = np.concatenate(X).astype(np.float32)
        Y = np.concatenate(Y).astype(np.float32)
        return {'X': X, 'Y':Y}




#########################
# tests
#########################

# # test Imagenet2012
# if __name__ == '__main__':
#     imagenet_dir = './datasets/imagenet'
#     dataName = 'imagenet2012_filtered.pkl'
#     imagenet_dataset = Imagenet2012(imagenet_dir)
#     imagenet_dataset.load_data()
#     imagenet_dataset.save_data_to_file(desName=dataName)
#     del imagenet_dataset

#     imagenet_dataset = Imagenet2012(imagenet_dir)
#     imagenet_dataset.load_data_from_file(os.path.join(imagenet_dataset.storage, dataName))
#     print(imagenet_dataset.size)
#     # imagenet_dataset.show_image(randomly=True)
#     del imagenet_dataset




# # test Imagenet2012_Val
# if __name__ == '__main__':
#     imagenet_dir = './datasets/imagenet'
#     dataName = 'imagenet2012_filtered.pkl'
#     imagenet_dataset = Imagenet2012(imagenet_dir)
#     imagenet_dataset.load_data_from_file(os.path.join(imagenet_dataset.storage, dataName))
#     label_dict = imagenet_dataset.label_dict
#     channel_mean = imagenet_dataset.channel_mean

#     val_imagenet_dir = './datasets/imagenet/validation_dataset'
#     val_dataName = 'imagenet2012_val.pkl'
#     val_imagenet_dataset = Imagenet2012_Val(val_imagenet_dir, label_dict, channel_mean)
#     val_imagenet_dataset.load_data()
#     val_imagenet_dataset.save_data_to_file(desName=val_dataName)
#     del val_imagenet_dataset

#     val_imagenet_dataset = Imagenet2012_Val(val_imagenet_dir, label_dict, channel_mean)
#     val_imagenet_dataset.load_data_from_file(os.path.join(val_imagenet_dataset.storage, val_dataName))
#     print(val_imagenet_dataset.size)
#     val_imagenet_dataset.show_image(randomly=True)




# # test sampler
# if __name__ == '__main__':
#     imagenet_dir = './datasets/imagenet'
#     dataName = 'imagenet2012.pkl'
#     dataset = Imagenet2012(imagenet_dir)
#     dataset.load_data_from_file(os.path.join(dataset.storage, dataName))

#     input_size = 224
#     channel_mean = dataset.channel_mean
#     num_cls = 1000
#     formater = Alexnet_Formater(input_size, channel_mean, num_cls)

#     rng = np.random.RandomState(seed=0)
#     batchsize = 1
#     sampler = Alexnet_Sampler(dataset, formater, batchsize)

#     ch = ord(' ')
#     while True:
#         train_data = sampler.sample_one(rng)
#         image = train_data['X'][0]
#         label = train_data['Y'][0]
#         if chr(ch) == 'n':
#             if label.argmax() != target_label:
#                 continue
#         image += dataset.channel_mean
#         image = image.astype(np.uint8)
#         print(image.shape)

#         windowName = 'show'
#         cv.namedWindow(windowName, cv.WINDOW_NORMAL)
#         cv.imshow(windowName, image)
#         ch = cv.waitKey()
#         if chr(ch) == 'q':
#             break
#         elif chr(ch) == 'n':
#             target_label = label.argmax()



# # test formater
# if __name__ == '__main__':
#     imagenet_dir = './datasets/imagenet'
#     dataName = 'imagenet2012.pkl'
#     dataset = Imagenet2012(imagenet_dir)
#     dataset.load_data_from_file(os.path.join(dataset.storage, dataName))

#     input_size = 224
#     channel_mean = dataset.channel_mean
#     num_cls = 1000
#     formater = Alexnet_Formater(input_size, channel_mean, num_cls)

#     data = dataset.data[dataset.data_ids[425858]]
#     imagePath = os.path.join(dataset.imageDir, data.imagePath)
#     image = cv.imread(imagePath)
#     coors = data.coors
#     label = data.label

#     formated = formater(image, coors, label)