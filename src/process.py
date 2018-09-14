# -*- coding:utf-8 -*-
#
#        Author : TangHanYi
#        E-mail : thydeyx@163.com
#   Create Date : 2018-09-12 18:08:55
# Last modified : 2018-09-12 18:09:00
#     File Name : process.py
#          Desc :

from itertools import islice

class Process(object):

    def __init__(self):
        pass

    def readFile(self, file_path):

        preId = 0
        data = []
        with open(file_path, 'r') as inf:
            for line in islice(inf, 1, None):
                line = line.strip().split()
                sentenceId = int(line[1])
                if sentenceId != preId:
                    try:
                        label = int(line[-1])
                        sentence = ' '.join(line[2:-1])
                        data.append((sentence, label))
                    except ValueError:
                        sentence = ' '.join(line[2:])
                        data.append(sentence)
                    finally:
                        preId = sentenceId

        return data