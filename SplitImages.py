# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:26:14 2020

@author: smoss
"""


import os
import shutil
first_round_training = './FirstRoundTraining'
second_round_training = './SecondRoundTraining'
snake_dir = './SnakePictures'
not_snake_dir = './NotSnakePictures'
snake_squares = '/Snakes'
not_snake_squares = '/NotSnakes'

def iterateOverDir(first_dir, second_dir, origin_dir, target_class):
    x = 0
    os.makedirs(first_round_training + target_class)
    os.makedirs(second_round_training + target_class)
    for file in os.listdir(origin_dir):
        copy_dir = first_dir
        if x % 10 == 0:
            copy_dir = second_dir
        shutil.copyfile(
            '{}/{}'.format(origin_dir, file),
            '{}{}/{}'.format(copy_dir, target_class, file)
        )
        x += 1
    print(target_class, x)

def createClasses():
    if os.path.exists(first_round_training):
        shutil.rmtree(first_round_training)
    if os.path.exists(second_round_training):
        shutil.rmtree(second_round_training)
    os.makedirs(first_round_training)
    os.makedirs(second_round_training)
    iterateOverDir(first_round_training, second_round_training, snake_dir, snake_squares)
    iterateOverDir(first_round_training, second_round_training, not_snake_dir, not_snake_squares)

if __name__ == "__main__":
    createClasses()