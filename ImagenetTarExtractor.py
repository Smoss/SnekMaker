import png
import os
# import pickle
import numpy
import tarfile
import argparse
from imagenetLabels import imagenet_labels, imagenet_original_labels
from PIL import Image
imagenet_dir = './ILSVRC2012_img_train'
initial_dir = './ImageNetImages'

def decodeDir(directory=imagenet_dir, target_dir=initial_dir):
    # labels = unpickle(imagenet_dir + '/batches.meta')
    
    tarballs = [x for x in os.listdir(directory)]
    x = 0
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for tarball_path in tarballs:
        file = '{}/{}'.format(directory, tarball_path)
        human_readable_label = imagenet_original_labels[tarball_path[:-4]]
        final_target = '{}/{}'.format(target_dir, human_readable_label)
        if not os.path.exists(final_target):
            os.makedirs(final_target)
        with tarfile.open(file) as tarball:
            tarball.extractall(final_target)
        print('Extract all the ' + human_readable_label + ' images to ' + final_target)
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode pickled images.')
    parser.add_argument(   
        '-d',
        '--directory',
        help='Directory to decode',
        default=imagenet_dir
    )
    parser.add_argument(   
        '-t',
        '--target',
        help='Directory to decode',
        default=initial_dir
    )
    args = parser.parse_args()
    decodeDir(args.directory, args.target)