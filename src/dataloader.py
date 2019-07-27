from __future__ import print_function
from .torchtools import *
import torch.utils.data as data
import random
import os
import numpy as np
from PIL import Image as pil_image
import pickle
from itertools import islice
from torchvision import transforms
import sys
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import argparse

"""Script to preprocess the omniglot dataset and pickle it into an array that's easy
    to index my character type"""


class Loader(data.Dataset):
    def __init__(self, root, partition='omniglot_background'):
        super(Loader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        self.data_size = [1, 84, 84]

        # set normalizer
        # mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        mean_pix = 120.39586422/255.0
        # std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        std_pix = 70.68188272/255.0
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'omniglot_background':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        self._save_to_pickle()
        # load data
        self.data = self.load_dataset()

    def _loadimgs(self):
    #if data not already unzipped, unzip it.
        path = self.root + '/' + self.partition
        if not os.path.exists(path):
            print("unzipping")
            os.chdir(data_path)
            os.system("unzip {}".format(path+".zip" ))

        data_dict = {}
        #we load every alphabet seperately so we can isolate them later
        for alphabet in os.listdir(path):
            print("loading alphabet: " + alphabet)
            print("Total number of alphabet: " + str(len(os.listdir(path))))
            alphabet_path = os.path.join(path,alphabet)
            print("Total number of character sets in "+alphabet+" is "+ str(len(os.listdir(alphabet_path))))
            #every letter/category has it's own column in the array, so  load seperately
            sum = 0;
            for character in os.listdir(alphabet_path):
                category_images=[]
                letter_path = os.path.join(alphabet_path, character)
                sum += len(os.listdir(letter_path))
                for filename in os.listdir(letter_path):
                    image_path = os.path.join(letter_path, filename)
                    image = imread(image_path)
                    category_images.append(image)

                data_dict[alphabet] = category_images
            print("Total number of images in "+alphabet+" is: " + str(sum))
        return data_dict

        #
    def _save_to_pickle(self):

        data=self._loadimgs()

        with open(os.path.join(self.root, self.partition + '.pickle'), "wb") as f:
            pickle.dump(data,f)

    def load_dataset(self):
        # load data
        dataset_path = os.path.join(self.root, self.partition+'.pickle')

        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)

        # for each class
        for c_idx in data:
            # for each image
            for i_idx in range(len(data[c_idx])):
                # resize
                image_data = pil_image.fromarray(np.uint8(data[c_idx][i_idx]))
                image_data = image_data.resize((self.data_size[2], self.data_size[1]))
                #image_data = np.array(image_data, dtype='float32')

                #image_data = np.transpose(image_data, (2, 0, 1))

                # save
                data[c_idx][i_idx] = image_data
        return data

    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1,
                       seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)


                # load sample for support set
                for i_idx in range(num_shots): 0 1
                    # set data
                    print(i_idx)
                    print(t_idx)
                    print(len(support_data))
                    support_data[i_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]