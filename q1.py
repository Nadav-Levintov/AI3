import csv

import math
import sklearn

flare_list = []
with open('flare.csv','r') as flare_file:
    flare_reader = csv.reader(flare_file,delimiter=' ', quotechar='|')
    for row in flare_reader:
        flare_list.append(row)

features = flare_list[0]
flare_list.remove(flare_list[0])

size =len(flare_list)
folds = [flare_list[x:x + math.ceil(size/4)] for x in range(0, len(flare_list), math.ceil(size/4))]