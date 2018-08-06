import torch
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader
import os

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, config):
        super(MnistDataLoader, self).__init__(config)
        self.data_dir = config['data_loader']['data_dir']
        self.data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=256, shuffle=False)
        self.x = []
        self.y = []
        for data, target in self.data_loader:
            self.x += [i for i in data.numpy()]
            self.y += [i for i in target.numpy()]
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __next__(self):
        batch = super(MnistDataLoader, self).__next__()
        batch = [np.array(sample) for sample in batch]
        return batch

    def _pack_data(self):
        packed = list(zip(self.x, self.y))
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.x, self.y = unpacked

    def _n_samples(self):
        return len(self.x)

class BLMPMDataLoader(BaseDataLoader):

    def __init__(self, config):
        
        data_dir = config['data_loader']['data_dir']

        self.word_embeddings = np.load(os.path.join(data_dir, 'word_embedding_matrix.npy'))

        if config['test'] :
            print("Loading test data...")
        else :
            print ("Loading training data...")

        q1_path = os.path.join(data_dir, 'q1_%s.npy' %('train_orig' if not config['test'] else 'test'))
        q2_path = os.path.join(data_dir, 'q2_%s.npy' %('train_orig' if not config['test'] else 'test'))
        
        self.p_sents = np.load(q1_path)
        self.q_sents = np.load(q2_path)

        
        self.x = list(zip(self.p_sents, self.q_sents))
        self.y = np.load(os.path.join(data_dir, 'label_train.npy'))
        
        self.config=config
        self.batch_size= config['data_loader']['batch_size']
        self.shuffle = config['data_loader']['shuffle']
        self.batch_idx = 0

    def __next__(self):
        """
        :return: Next batch
        """
        packed = self._pack_data()
        if self.batch_idx < self.__len__():
            batch = packed[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
            self.batch_idx = self.batch_idx + 1
            x, y=  self._unpack_data(batch)
            p, q = list(zip(*x))
            return p, q, y

        else:
            raise StopIteration

    def _pack_data(self):
        packed = list(zip(self.x, self.y))
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.x, self.y = unpacked

    def _n_samples(self):
        return len(self.x)

    def get_pretrained_embeddings(self):
        return self.word_embeddings

