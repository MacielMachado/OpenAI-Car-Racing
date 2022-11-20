import numpy as np

class DataHandler():
    def __init__(self):
        pass

    def load_data(self, path):
        return np.load(path).astype('float32')

    def append_data(self, array_1, array_2):
        return np.append(array_1, array_2, axis=0)

    def frac_array(self, array, frac):
        length = len(array)
        return np.array(array[:int((1-frac) * length)]),\
               np.array(array[int((1-frac) * length):])

    def to_greyscale(self, imgs):
        return np.dot(imgs, [0.2989, 0.5870, 0.1140])
        
    def normalizing(self, imgs):
        return imgs/255.0