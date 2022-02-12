import os
import shutil


DEST = 'VALIDATION_DATA/'
SRC = "DATA/"
FILENAME = 'DATA/validation_list.txt'


with open(FILENAME, 'r') as fin:
        for line in fin:
            # print(line.strip().split('/')[0])
            if not os.path.exists(DEST + line.strip().split('/')[0]):
                os.makedirs(DEST + line.strip().split('/')[0])
            shutil.move(SRC + line.strip(), DEST + line.strip())





