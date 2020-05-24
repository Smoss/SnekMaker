import png
import os
import pickle
import numpy
import itertools
import argparse
from PIL import Image
cifar_10_dir = './Cifar10Data'
initial_dir = './NotSnakePictures'
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def decodeDir(directory=cifar_10_dir, target_dir=initial_dir):
    # labels = unpickle(cifar_10_dir + '/batches.meta')
    
    files = [x for x in os.listdir(directory) if (x.startswith('data') or x.startswith('test') or x.startswith('train'))]
    x = 0
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file in files:
        file = '{}/{}'.format(directory, file)
        cifar_data = unpickle(file)
        print(cifar_data.keys())
        pics = cifar_data[b'data']
        names = cifar_data[b'filenames']
        labels = cifar_data[b'labels']
        zipped = zip(pics, names, labels)
        for pixels, name, label in zipped:
            image_3d = numpy.reshape(pixels, (3, 32, 32)).transpose([1, 2, 0])
            temp_image = Image.fromarray(image_3d)
            # print(str(name,'utf-8'))
            # print(str(numpy.reshape(image_3d, (-1, 32*3)).shape))
            target_folder = '{}/{}'.format(target_dir, label)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            temp_image.save(target_folder + '/' + str(name,'utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode pickled images.')
    parser.add_argument(   
        '-d',
        '--directory',
        help='Directory to decode'
    )
    parser.add_argument(   
        '-t',
        '--target',
        help='Directory to decode'
    )
    args = parser.parse_args()
    decodeDir(args.directory, args.target)