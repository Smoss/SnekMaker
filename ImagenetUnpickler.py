import png
import os
import pickle
import numpy
import itertools
import argparse
from imagenetLabels import imagenet_labels
from PIL import Image
imagenet_dir = './ImageNet'
initial_dir = './ImageNetImages'
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict

def decodeDir(directory=imagenet_dir, target_dir=initial_dir):
    # labels = unpickle(imagenet_dir + '/batches.meta')
    
    files = [x for x in os.listdir(directory) if (x.startswith('data') or x.startswith('test') or x.startswith('train'))]
    x = 0
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file in files:
        file = '{}/{}'.format(directory, file)
        cifar_data = unpickle(file)
        # print(cifar_data.keys())
        pics = cifar_data['data']
        # names = cifar_data[b'filenames']
        labels = cifar_data['labels']
        zipped = zip(pics, labels)
        index = 0
        for pixels, label in zipped:
            image_3d = numpy.reshape(pixels, (3, 64, 64)).transpose([1, 2, 0])
            temp_image = Image.fromarray(image_3d)
            # print(str(name,'utf-8'))
            # print(str(numpy.reshape(image_3d, (-1, 32*3)).shape))
            human_readable_label = imagenet_labels[label][0]
            target_folder = '{}/{}'.format(target_dir, human_readable_label)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            temp_image.save(target_folder + '/' + str(
                '{}_{}'.format(human_readable_label, index)
            ) + '.png')
            index += 1

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