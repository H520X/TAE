# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F
from torch import optim
import math
import numpy as np
import scipy.sparse as sp
from utils.Args import args
from utils.dataset import Dataset
from model.TAE_ConvE import tam_conve
from utils.measure import *
from utils.util import *
import random
import time
from tqdm import tqdm
import warnings
import gc

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.enabled = False
cudnn.benchmark = False
cudnn.deterministic = True

warnings.filterwarnings(action='ignore')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# 数据集加载
data_loader = Dataset(args.dataset)

if args.dataset == 'ICEWS14':
    quadrupleList_t_train, timestampsList_train, adj_t_train = data_loader.get_quadruple_with_t('train.txt')
    quadrupleList_t_valid, timestampsList_valid, adj_t_valid = data_loader.get_quadruple_with_t('test.txt')

    all_quadrupleList_t = quadrupleList_t_train + quadrupleList_t_valid

else:
    quadrupleList_t_train, timestampsList_train, adj_t_train = data_loader.get_quadruple_with_t('train.txt')
    quadrupleList_t_valid, timestampsList_valid, adj_t_valid = data_loader.get_quadruple_with_t('valid.txt')
    quadrupleList_t_test, timestampsList_test, adj_t_test = data_loader.get_quadruple_with_t('test.txt')

    all_quadrupleList_t = quadrupleList_t_train + quadrupleList_t_valid + quadrupleList_t_test

print('dataset dataload succeed')

all_entityList, all_relationList = data_loader.get_entityAndRlationList(all_quadrupleList_t)
all_entityList.sort()
all_relationList.sort()
ent_num = max(all_entityList) + 1
rel_num = max(all_relationList) + 1

print('dataset process succeed')

all_quadrupleList_t = []

batch_size_conv = args.batch_size_conv

all_quadrupleList = []
all_timestampsList = []

if args.pred == 'sub':
    mkdirs('./results_new/sub/ConvE/{}'.format(args.dataset))
    model_state_file = "./results_new/sub/ConvE/{}/model_state.pth".format(args.dataset)
elif args.pred == 'obj':
    mkdirs('./results_new/obj/ConvE/{}'.format(args.dataset))
    model_state_file = "./results_new/obj/ConvE/{}/model_state.pth".format(args.dataset)
else:
    mkdirs('./results_whole/rel/ConvE/{}'.format(args.dataset))
    model_state_file = "./results_whole/rel/ConvE/{}/model_state.pth".format(args.dataset)

def train(args):
    print('start train')
    # 模型加载

    count = 0
    best_mrr = 0
    best_hits1 = 0
    best_hits3 = 0
    best_hits10 = 0

    model = tam_conve(args, ent_num, rel_num)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_conv, amsgrad=True)

    if os.path.exists(model_state_file):
        checkpoint = torch.load(model_state_file)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # print(optimizer.param_groups[0]['lr'])
        # optimizer.param_groups[0]['lr'] = 0.0001
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, args.n_epochs_conv)):
        train_loss = 0

        for time_idx in tqdm(range(len(timestampsList_train))):
            model.train()

            train_quadruple_list = quadrupleList_t_train[time_idx]
            timeList = [timestampsList_train[time_idx]]

            train_quadruple = np.array(train_quadruple_list)
            n_batch = (train_quadruple.shape[0] + batch_size_conv - 1) // batch_size_conv

            for idx in range(n_batch):
                out_ent_embeds, out_rel_embeds = model.get_temporal_entity_And_relation_Embed_with_t(timeList)
                out_ent_embeds = out_ent_embeds.cuda()
                out_rel_embeds = out_rel_embeds.cuda()

                batch_start = idx * batch_size_conv
                batch_end = min(train_quadruple.shape[0], (idx + 1) * batch_size_conv)

                train_batch_quadruple = train_quadruple[batch_start: batch_end, :]  # 将每个时间戳下的所有三元组分批训练

                if args.pred == 'sub':
                    train_targets = torch.Tensor(train_batch_quadruple[:, 0]).long().to('cuda')  # 生成每个batch的标签
                    train_scores = model.forward(out_ent_embeds, out_rel_embeds, train_batch_quadruple, args.pred)
                    loss = model.loss(train_scores, train_targets) + model.regularization_loss()

                elif args.pred == 'obj':
                    train_targets = torch.Tensor(train_batch_quadruple[:, 2]).long().to('cuda')
                    train_scores = model.forward(out_ent_embeds, out_rel_embeds, train_batch_quadruple, args.pred)
                    loss = model.loss(train_scores, train_targets) + model.regularization_loss()

                else:
                    train_targets = torch.Tensor(train_batch_quadruple[:, 1]).long().to('cuda')
                    train_scores = model.forward(out_ent_embeds, out_rel_embeds, train_batch_quadruple, args.pred)
                    loss = model.loss(train_scores, train_targets) + model.regularization_loss()

                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()

        print('each_epoch_loss:', train_loss)

        if epoch < args.valid_epoch:
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                       model_state_file)

        if epoch >= args.valid_epoch:
            model.eval()

            test_mrr, test_hits1, test_hits3, test_hits10 = 0, 0, 0, 0
            num = 0
            with torch.no_grad():
                if args.multi_step:
                    for time_idx in tqdm(range(len(timestampsList_test))):
                        test_quadruple_list = quadrupleList_t_test[time_idx]

                        num += len(test_quadruple_list)

                        timeList = [timestampsList_test[time_idx]]

                        test_quadruple = np.array(test_quadruple_list)
                        n_batch = (test_quadruple.shape[0] + batch_size_conv - 1) // batch_size_conv

                        out_ent_embeds, out_rel_embeds = model.get_temporal_entity_And_relation_Embed_with_t(timeList)
                        out_ent_embeds = out_ent_embeds.cuda()
                        out_rel_embeds = out_rel_embeds.cuda()

                        for idx in range(n_batch):
                            batch_start = idx * batch_size_conv
                            batch_end = min(test_quadruple.shape[0], (idx + 1) * batch_size_conv)

                            test_batch_quadruple = test_quadruple[batch_start: batch_end, :]  # 将每个时间戳下的所有三元组分批训练

                            if args.pred == 'sub':
                                test_targets = torch.Tensor(test_batch_quadruple[:, 0]).long().to('cuda')
                                test_scores = model.forward(out_ent_embeds, out_rel_embeds, test_batch_quadruple, args.pred)

                            elif args.pred == 'obj':
                                test_targets = torch.Tensor(test_batch_quadruple[:, 2]).long().to('cuda')
                                test_scores = model.forward(out_ent_embeds, out_rel_embeds, test_batch_quadruple, args.pred)

                            else:
                                test_targets = torch.Tensor(test_batch_quadruple[:, 1]).long().to('cuda')
                                test_scores = model.forward(out_ent_embeds, out_rel_embeds, test_batch_quadruple, args.pred)

                            temp_mrr, temp_hits1, temp_hits3, temp_hits10 = get_performanceIndex(test_scores, test_targets)

                            test_mrr += temp_mrr * len(test_batch_quadruple)
                            test_hits1 += temp_hits1 * len(test_batch_quadruple)
                            test_hits3 += temp_hits3 * len(test_batch_quadruple)
                            test_hits10 += temp_hits10 * len(test_batch_quadruple)

                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()
                        torch.cuda.empty_cache()

                test_mrr = test_mrr / num
                test_hits1 = test_hits1 / num
                test_hits3 = test_hits3 / num
                test_hits10 = test_hits10 / num

            print("\n MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f}".
                  format(test_mrr, test_hits1, test_hits3, test_hits10))

            if test_mrr > best_mrr:
                best_mrr = test_mrr

                count = 0

                torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                           model_state_file)
            else:
                count += 1

            if test_hits1 > best_hits1:
                best_hits1 = test_hits1
            if test_hits3 > best_hits3:
                best_hits3 = test_hits3
            if test_hits10 > best_hits10:
                best_hits10 = test_hits10

            print(
                "Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f}".
                    format(epoch + 1, train_loss, best_mrr, best_hits1, best_hits3, best_hits10))

            if count == args.count:
                break

train(args)
print('train done')

