"""
    This file splits images into a train and a test set
    Copyright (C) 2020  Fabian Kahl, alias Fabian Phatrong  phatrong@web.de

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
import glob
import random
import shutil

# directories and classes
filelocation = '/media/phatrong/Data/Uni/Masterthesis-WS1920/Images/'
classes = ['Elephant/', 'Platypus/', 'Unicorn/']
destination_train = '/media/phatrong/Data/Uni/Masterthesis-WS1920/Images/train/'
destination_test = '/media/phatrong/Data/Uni/Masterthesis-WS1920/Images/test/'

# percentage of data to be in test set
chance_of_test = 0.1

# if train and test directories do not exist, then create them
if not os.path.exists(destination_train):
    os.makedirs(destination_train)
if not os.path.exists(destination_test):
    os.makedirs(destination_test)

# run over all classes
for j in classes:

    # if class directories do not exist, then create them
    if not os.path.exists(destination_train + j):
        os.makedirs(destination_train + j)
    if not os.path.exists(destination_test + j):
        os.makedirs(destination_test + j)

    # load all images
    data_path = os.path.join(filelocation + j,'*.jpg')
    files = glob.glob(data_path)

    # run over all images
    for i in range(len(files)):
        
        # load image i
        img_loc = files[i]
        img_file_name = os.path.basename(img_loc)
        
        # generate a random number between 0. and 1.
        random_number = random.random()
        
        # sort image in train or test directories
        if random_number > chance_of_test:
            shutil.copy(img_loc, destination_train + j + img_file_name)
        else:
            shutil.copy(img_loc, destination_test + j + img_file_name)