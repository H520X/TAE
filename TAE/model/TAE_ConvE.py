# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from utils.Args import args
import time
import math

CUDA = torch.cuda.is_available()


class tam_conve(torch.nn.Module):
    def __init__(self, args, ent_num, rel_num):

        super(tam_conve, self).__init__()

        # 编码器
        self.embedding_dim = args.embedding_dim
        self.embedding_dim1 = args.embedding_dim1
        self.embedding_dim2 = args.embedding_dim // args.embedding_dim1

        self.ent_num = ent_num
        self.rel_num = rel_num

        # 获取实体、关系、时间戳的嵌入权重矩阵
        self.entity_init_embeds = nn.Parameter(torch.Tensor(self.ent_num, self.embedding_dim), requires_grad=True)
        self.relation_init_embeds = nn.Parameter(torch.Tensor(self.rel_num, self.embedding_dim), requires_grad=True)
        self.timestamp_init_embeds = nn.Parameter(torch.Tensor(1, self.embedding_dim), requires_grad=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()  # inplace=False)
        self.tanh = nn.Tanh()

        self.drop_ta = nn.Dropout(args.dropout_ta)

        self.w_t_fea = nn.Parameter(torch.zeros(size=(self.embedding_dim, self.embedding_dim)))
        self.w_t_ent = nn.Parameter(torch.zeros(size=(self.ent_num, 1)))

        self.w_t_fea_r = nn.Parameter(torch.zeros(size=(self.embedding_dim, self.embedding_dim)))
        self.w_t_rel = nn.Parameter(torch.zeros(size=(self.rel_num, 1)))

        self.inp_drop = torch.nn.Dropout(args.input_drop, inplace=False)
        self.hidden_drop = torch.nn.Dropout(args.hidden_drop, inplace=False)
        self.feature_drop = torch.nn.Dropout2d(args.feat_drop, inplace=False)

        self.conv = nn.Conv2d(1, 32, 4, padding=1)

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm1d(args.embedding_dim)
        self.bn_conv = nn.BatchNorm2d(32)

        self.fc = nn.Linear(args.hidden_dim, args.embedding_dim)
        self.loss = nn.CrossEntropyLoss()

        self.reg_para = args.reg_para

        self.reset_parameters()

    # 编码器函数
    def reset_parameters(self):  # 初始化实体、关系、时间戳的嵌入权重矩阵
        nn.init.xavier_uniform_(self.entity_init_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.relation_init_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.timestamp_init_embeds, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.w_t_fea, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_t_ent, gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.w_t_fea_r, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_t_rel, gain=nn.init.calculate_gain('relu'))

    def get_init_timesatmp(self, timeList):  # 获取所有时间戳的嵌入矩阵，输入时间戳是时间戳列表
        init_timestamps = torch.Tensor(len(timeList), self.embedding_dim)

        for i in range(len(timeList)):
            init_timestamps[i] = torch.Tensor(
                self.timestamp_init_embeds.cpu().detach().numpy().reshape(self.embedding_dim)) * (
                                             timeList[i] // args.time_interval + 1)
            # self.timestamp_init_embeds.cpu().detach().numpy().reshape(self.embedding_dim)) * (i + 1)

        init_timestamps = init_timestamps.to('cuda')

        return init_timestamps

    def get_temporal_entity_And_relation_Embed_with_t(self, timeList):
        init_timestamps = self.get_init_timesatmp(timeList)

        # init_entity = self.entity_init_embeds.clone().to('cuda')  # 获取每个时间戳下的所有实体的初始嵌入矩阵
        # init_relation = self.relation_init_embeds.clone().to('cuda')  # 获取每个时间戳下的所有实体的初始嵌入矩阵
        timestamp = timeList[-1] // args.time_interval

        # 时间感知注意力
        entity_embed_t = torch.matmul(init_timestamps[-1], self.w_t_fea).float()
        impact_factor_e = F.softmax(entity_embed_t, dim=-1)
        impact_factor_e_mat = impact_factor_e.repeat(self.ent_num, 1)
        impact_factor_e_mat = self.w_t_ent * impact_factor_e_mat

        out_ent_embeds = self.relu(self.drop_ta(
            torch.mul(self.entity_init_embeds * (timestamp + 1), impact_factor_e_mat))) + self.entity_init_embeds

        relation_embed_t = torch.matmul(init_timestamps[-1], self.w_t_fea_r).float()
        impact_factor_r = F.softmax(relation_embed_t, dim=-1)
        impact_factor_r_mat = impact_factor_r.repeat(self.rel_num, 1)
        impact_factor_r_mat = self.w_t_rel * impact_factor_r_mat

        out_rel_embeds = self.relu(self.drop_ta(
            torch.mul(self.relation_init_embeds * (timestamp + 1),
                      impact_factor_r_mat))) + self.relation_init_embeds

        return out_ent_embeds, out_rel_embeds

    def circonv(self, input_matrix):
        x = input_matrix[:, :, :, :1]
        output_matrix = torch.cat((input_matrix, x), dim=3)
        y = output_matrix[:, :, :1, :]
        temp_matrix = output_matrix.clone()
        output_matrix = torch.cat((temp_matrix, y), dim=2)

        return output_matrix

    def forward(self, ent_embeds, rel_embeds, quadruples, pred):
        if pred == 'sub':  # 头实体预测
            tailList = quadruples[:, 2]
            relList = quadruples[:, 1]
            tail_embedding = ent_embeds[tailList].to('cuda')
            rel_embedding = rel_embeds[relList].to('cuda')

            head_embedding = ent_embeds.to('cuda')

            tail_embed = tail_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            rel_embed = rel_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            input_matrix = self.inp_drop(self.bn0(torch.cat((tail_embed, rel_embed), 2)))
            feature_matrix = self.feature_drop(self.relu(self.bn_conv(self.conv(self.circonv(input_matrix)))))
            fc_vector = self.relu(
                self.bn1(self.hidden_drop(self.fc(feature_matrix.view(feature_matrix.shape[0], -1)))))
            score = torch.mm(fc_vector, head_embedding.transpose(1, 0))

        elif pred == 'obj':  # 尾实体预测
            headList = quadruples[:, 0]
            relList = quadruples[:, 1]
            head_embedding = ent_embeds[headList].to('cuda')
            rel_embedding = rel_embeds[relList].to('cuda')

            tail_embedding = ent_embeds.to('cuda')

            head_embed = head_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            rel_embed = rel_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            input_matrix = self.inp_drop(self.bn0(torch.cat((head_embed, rel_embed), 2)))
            feature_matrix = self.feature_drop(self.relu(self.bn_conv(self.conv(self.circonv(input_matrix)))))
            fc_vector = self.relu(
                self.bn1(self.hidden_drop(self.fc(feature_matrix.view(feature_matrix.shape[0], -1)))))
            score = torch.mm(fc_vector, tail_embedding.transpose(1, 0))

        else:  # 关系预测
            headList = quadruples[:, 0]
            tailList = quadruples[:, 2]
            head_embedding = ent_embeds[headList].to('cuda')
            tail_embedding = ent_embeds[tailList].to('cuda')

            rel_embedding = rel_embeds.to('cuda')

            head_embed = head_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            tail_embed = tail_embedding.view(-1, 1, self.embedding_dim1, self.embedding_dim2)
            input_matrix = self.inp_drop(self.bn0(torch.cat((head_embed, tail_embed), 2)))
            feature_matrix = self.feature_drop(self.relu(self.bn_conv(self.conv(self.circonv(input_matrix)))))
            fc_vector = self.relu(
                self.bn1(self.hidden_drop(self.fc(feature_matrix.view(feature_matrix.shape[0], -1)))))
            score = torch.mm(fc_vector, rel_embedding.transpose(1, 0))

        return score

    def regularization_loss(self):
        regularization_loss = torch.mean(self.entity_init_embeds.pow(2)) + torch.mean(
            self.relation_init_embeds.pow(2)) + torch.mean(self.timestamp_init_embeds.pow(2))
        return regularization_loss * self.reg_para



























































































