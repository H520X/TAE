import os
import torch
import torch.nn as nn
import numpy as np


class Dataset:
    def __init__(self, dataset_name):
        self.name = dataset_name
        self.dataset_path = "../data/" + dataset_name + "/"

    def get_quadruple_with_t(self, filename):  # 存储每个数据集下的所有三元组
        quadrupleList = []  # 用列表存储每一个事实四元组[(头实体、关系、尾实体、时间戳)]
        timestampsList = []
        quadrupleList_t = []  # 用列表存储每个时间戳下的四元组列表[[]...[]]

        adj_t = []

        with open(os.path.join(self.dataset_path, filename), 'r') as fr:

            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                relation = int(line_split[1])
                timestamp = int(line_split[3])
                quadruple = (head, relation, tail, timestamp)

                quadrupleList.append(quadruple)


                if timestamp not in timestampsList:
                    timestampsList.append(timestamp)  # 单独构建一个存储所有时间戳的列表

        for timestamp in timestampsList:
            temp_quadrupleList = []
            rows, cols, data = [], [], []

            for quadruple in quadrupleList:
                if quadruple[3] == timestamp:
                    temp_quadrupleList.append(quadruple)

                    rows.append(quadruple[0])
                    cols.append(quadruple[2])
                    data.append(quadruple[1])
                    rows.append(quadruple[2])
                    cols.append(quadruple[0])
                    data.append(quadruple[1])

            quadrupleList_t.append(temp_quadrupleList)
            adj_t.append((rows, cols, data))

        return quadrupleList_t, timestampsList, adj_t

    def get_entityAndRlationList(self,quadrupleList_t):
        all_entityList = []
        all_relationList = []
        for i in range(len(quadrupleList_t)):
            for quadruple in quadrupleList_t[i]:
                for entity in [quadruple[0], quadruple[2]]:
                    if entity not in all_entityList:
                        all_entityList.append(entity)
                if quadruple[1] not in all_relationList:
                    all_relationList.append(quadruple[1])

        return all_entityList, all_relationList






























