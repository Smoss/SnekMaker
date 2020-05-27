import os
import argparse
import shutil
import csv
import random
imagenet_dir = './ImageNetImages'
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

def decodeDir(
        directory=imagenet_dir,
        only_snakes=False
    ):
    train_csv_rows = []
    validate_csv_rows = []
    # labels = unpickle(cifar_10_dir + '/batches.meta')
    labels = [x for x in os.listdir(directory)]
    for label in labels:
        validator_splitter = 0
        from_dir = '{}/{}'.format(directory, label)
        for file in os.listdir(from_dir):
            from_path = '{}/{}'.format(from_dir, file)
            abs_path = os.path.abspath(from_path)
            is_snake = label in snake_labels
            target_class = '[1]' if is_snake else '[0]'
            row_dict = {'file': abs_path, 'snake': target_class}
            if not only_snakes or (is_snake or label.startswith('fake_snake')):
                if validator_splitter % 10 == 0:
                    validate_csv_rows.append(row_dict)
                else:
                    train_csv_rows.append(row_dict)
                validator_splitter += 1
        # print('Finished handling', label)
    headers = ['file', 'snake']
    # print(len(train_csv_rows))
    # print(len(validate_csv_rows))
    random.shuffle(train_csv_rows)
    random.shuffle(validate_csv_rows)
    with open('classes_train.csv', 'w') as train_file:       
        train_csv = csv.DictWriter(train_file, headers)
        train_csv.writeheader()
        train_csv.writerows(train_csv_rows)

    with open('classes_validate.csv', 'w') as validate_file:       
        validate_csv = csv.DictWriter(validate_file, headers)
        validate_csv.writeheader()
        validate_csv.writerows(validate_csv_rows)
    
    print('Sifted all the data')
    return len(train_csv_rows), len(validate_csv_rows)

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
        '--only-snakes',
        help='Directory to decode',
        action='store_true'
    )
    args = parser.parse_args()
    decodeDir(
        args.directory,
        only_snakes=args.only_snakes
    )