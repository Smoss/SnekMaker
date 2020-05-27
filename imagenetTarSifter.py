import png
import os
# import pickle
import numpy
import tarfile
import argparse
from imagenetLabels import imagenet_labels, imagenet_original_labels
from PIL import Image
imagenet_dir = './ILSVRC2012_img_train'
snake_dir = './SnakePictures'
not_snake_dir = './NotSnakePictures'
snake_labels = [
    'Indian_cobra',
    'boa_constrictor',
    'diamondback',
    'garter_snake',
    'green_mamba',
    'green_snake',
    'hognose_snake',
    'horned_viper',
    'king_snake',
    'night_snake',
    'ringneck_snake',
    'rock_python',
    'sidewinder',
    'thunder_snake',
    'vine_snake',
    'water_snake'
]

def decodeDir(directory=imagenet_dir, snake_target_dir=snake_dir, not_snake_target_dir=not_snake_dir):
    # labels = unpickle(imagenet_dir + '/batches.meta')
    
    tarballs = [x for x in os.listdir(directory)]
    x = 0
    if not os.path.exists(snake_target_dir):
        os.makedirs(snake_target_dir)
    if not os.path.exists(not_snake_target_dir):
        os.makedirs(not_snake_target_dir)
    for tarball_path in tarballs:
        file = '{}/{}'.format(directory, tarball_path)
        human_readable_label = imagenet_original_labels[tarball_path[:-4]]
        target_dir = snake_target_dir if human_readable_label in snake_labels else not_snake_target_dir
        final_target = '{}'.format(target_dir)
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
        '-s',
        '--snakes',
        help='Directory to decode',
        default=snake_dir
    )
    parser.add_argument(   
        '-n',
        '--not-snakes',
        help='Directory to decode',
        default=not_snake_dir
    )
    args = parser.parse_args()
    decodeDir(args.directory, args.snakes, args.not_snakes)