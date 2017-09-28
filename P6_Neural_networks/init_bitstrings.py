# @Author: Patrice Bechard <Patrice>
# @Date:   2017-03-05T15:35:18-05:00
# @Email:  bechardpatrice@gmail.com
# @Last modified by:   Patrice
# @Last modified time: 2017-04-12T18:26:21-04:00

import numpy as np

nSample = 1000
lenUnit = 12

files = ['bitstrings_train.txt','bitstrings_test.txt']
for file in files:
    f = open(file,'w')
    f.write(str(nSample)+'\n')
    numbad = 0
    numgood = 0
    while numgood != nSample//2:
        string = ''
        for j in range(lenUnit):
            string += str(np.random.randint(2))
        count = 0
        good = False
        for char in string:
            if int(char) == 1:
                count += 1
                if count == 5:
                    good = True
                    break
            else:
                count = 0
        if good:
            string += ' 1'
            numgood += 1
        else:
            string += ' 0'
            numbad += 1
            if numbad > nSample // 2:
                continue
        string += '\n'
        f.write(string)
