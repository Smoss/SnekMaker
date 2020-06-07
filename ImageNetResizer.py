import png
import os
# import pickle
import numpy
import tarfile
import argparse
from imagenetLabels import imagenet_labels, imagenet_original_labels, skip_labels
from PIL import Image

# imagenet_dir = './ILSVRC2012_img_train'
INITIAL_DIR = './ImageNetImagesUnsized'
TARGET_DIR = './ImageNetImages'
TARGET_SIZE = 128

def decodeDir(directory=INITIAL_DIR, target_dir=TARGET_DIR, target_size=TARGET_SIZE):
    train_csv_rows = []
    validate_csv_rows = []
    for label in os.listdir(directory):
        if label in skip_labels:
            continue
        validator_splitter = 0
        from_dir = '{}/{}'.format(directory, label)
        to_dir = '{}/{}'.format(target_dir, label)
        if not os.path.exists(to_dir):
            os.makedirs(to_dir)
        for file in os.listdir(from_dir):
            from_path = '{}/{}'.format(from_dir, file)
            # target_name = file[:-3] + 'bmp'
            target_path = '{}/{}'.format(to_dir, file)
            im = Image.open(from_path)
            target_im = im.resize((target_size, target_size), resample=Image.BICUBIC)
            try:
                target_im.save(target_path)
            except:
                print(target_path)
            # abs_path = os.path.abspath(from_path)
            # row_dict = {'file': abs_path, 'class_num': [target_class]}
            # if validator_splitter % 10 == 0:
            #     validate_csv_rows.append(row_dict)
            # else:
            # train_csv_rows.append(row_dict)
            # validator_splitter += 1
        print('Finished handling', label)

    # print('Sifted all the data')
    # print(len(train_csv_rows), len(validate_csv_rows))
    # return len(train_csv_rows), len(validate_csv_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode pickled images.')
    parser.add_argument(
        '-d',
        '--directory',
        help='Directory to decode',
        default=INITIAL_DIR
    )
    parser.add_argument(
        '-t',
        '--target',
        help='Directory to decode',
        default=TARGET_DIR
    )
    args = parser.parse_args()
    decodeDir(args.directory, args.target)