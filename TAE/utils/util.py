# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import os
import time

def sample(batch_indices, all_indices, pred, ent_num, rel_num):
    batch_num = len(batch_indices)
    batch_indices_with_neg = np.tile(batch_indices, (2, 1))

    for i in range(batch_num):
        if pred == 'sub':
            while(batch_indices_with_neg[batch_num+i, 0], batch_indices_with_neg[batch_num+i, 1],
                             batch_indices_with_neg[batch_num+i, 2], batch_indices_with_neg[batch_num+i, 3]) in all_indices:
                batch_indices_with_neg[batch_num + i, 0] = np.random.randint(0, ent_num)

        elif pred == 'obj':
            while(batch_indices_with_neg[batch_num+i, 0], batch_indices_with_neg[batch_num+i, 1],
                             batch_indices_with_neg[batch_num+i, 2], batch_indices_with_neg[batch_num+i, 3]) in all_indices:
                batch_indices_with_neg[batch_num + i, 2] = np.random.randint(0, ent_num)

        else:
            while (batch_indices_with_neg[batch_num + i, 0], batch_indices_with_neg[batch_num + i, 1],
                             batch_indices_with_neg[batch_num+i, 2], batch_indices_with_neg[batch_num+i, 3]) in all_indices:
                batch_indices_with_neg[batch_num + i, 1] = np.random.randint(0, rel_num)

    return batch_indices_with_neg

def construct_snap(test_quadruples, final_score, topK, pred):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_quadruples = []

    rows, cols, data = [], [], []
    
    if pred == 'sub':
        for _ in range(len(test_quadruples)):
            for index in top_indices[_]:
                t, r = test_quadruples[_][2], test_quadruples[_][1]
                predict_quadruples.append((index, r, t))
                
    elif pred == 'obj':
        for _ in range(len(test_quadruples)):
            for index in top_indices[_]:
                h, r = test_quadruples[_][0], test_quadruples[_][1]
                predict_quadruples.append((h, r, index))
                
    else:
        for _ in range(len(test_quadruples)):
            for index in top_indices[_]:
                h, t = test_quadruples[_][0], test_quadruples[_][2]
                predict_quadruples.append((h, index, t))

    for quadruple in predict_quadruples:
        rows.append(quadruple[0])
        cols.append(quadruple[2])
        data.append(quadruple[1])
        rows.append(quadruple[2])
        cols.append(quadruple[0])
        data.append(quadruple[1])

    return rows, cols, data

def get_validation_pred_ent(model, ent_embeds, rel_embeds, quads, entity_list):
    sub_mr, obj_mr = 0, 0
    sub_mrr, obj_mrr = 0, 0
    sub_h10, obj_h10 = 0, 0
    sub_h3, obj_h3 = 0, 0
    sub_h1, obj_h1 = 0, 0

    for iters in range(1):
        indices = [i for i in range(len(quads))]
        batch_indices = quads[indices, :]

        for i in range(batch_indices.shape[0]):
            new_x_batch_head = np.tile(batch_indices[i, :], (len(entity_list), 1))
            new_x_batch_tail = np.tile(batch_indices[i, :], (len(entity_list), 1))

            new_x_batch_head[:, 0] = entity_list    #头实体预测
            new_x_batch_tail[:, 2] = entity_list    #尾实体预测

            # adding the current valid triples to the top, i.e, index 0
            new_x_batch_head = np.insert(
                new_x_batch_head, 0, batch_indices[i], axis=0)   #把正确的这个三元组放在第一行，下面的全是错误的
            new_x_batch_tail = np.insert(
                new_x_batch_tail, 0, batch_indices[i], axis=0)   #同上

            import math
            # Have to do this, because it doesn't fit in memory

            scores_head = model.forward(ent_embeds, rel_embeds, torch.LongTensor((new_x_batch_head)).cuda())

            sorted_scores_head, sorted_indices_head = torch.sort(scores_head.view(-1), dim=-1, descending=True)
            # Just search for zeroth index in the sorted scores, we appended valid triple at top
            sub_rank = np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1
            sub_mr += sub_rank
            sub_mrr += 1.0 / sub_rank

            # Tail part here
            scores_tail = model.forward(ent_embeds, rel_embeds, torch.LongTensor((new_x_batch_tail)).cuda())

            sorted_scores_tail, sorted_indices_tail = torch.sort(scores_tail.view(-1), dim=-1, descending=True)

            # Just search for zeroth index in the sorted scores, we appended valid triple at top
            obj_rank = np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1
            obj_mr += obj_rank
            obj_mrr += 1.0 / obj_rank


            if sub_rank <= 10:
                sub_h10 += 1
            if sub_rank <= 3:
                sub_h3 += 1
            if sub_rank == 1:
                sub_h1 += 1

            if obj_rank <= 10:
                obj_h10 += 1
            if obj_rank <= 3:
                obj_h3 +=1
            if obj_rank == 1:
                obj_h1 += 1

    return sub_mrr + obj_mrr, sub_h1 + obj_h1, sub_h3 + obj_h3, sub_h10 + obj_h10


def get_validation_pred_rel(model, quads, relation_list):
    mr, mrr, h1, h3, h10 = 0,0,0,0,0

    for iters in range(1):
        indices = [i for i in range(len(quads))]
        batch_indices = quads[indices, :]

        for i in range(batch_indices.shape[0]):
            new_x_batch_rel = np.tile(batch_indices[i, :], (len(relation_list), 1))

            if (batch_indices[i, 1]) not in unique_relations:
                continue  # 保证测试集里面的关系都是训练集里出现过的

            new_x_batch_rel[:, 1] = relation_list  # 关系预测

            # adding the current valid triples to the top, i.e, index 0
            new_x_batch_rel = np.insert(
                new_x_batch_rel, 0, batch_indices[i], axis=0)  # 把正确的这个三元组放在第一行，下面的全是错误的

            import math
            # Have to do this, because it doesn't fit in memory

            scores_rel = model.forward(ent_embeds, rel_embeds, torch.LongTensor((new_x_batch_rel)).cuda())

            sorted_scores_rel, sorted_indices_rel = torch.sort(scores_rel.view(-1), dim=-1, descending=True)
            # Just search for zeroth index in the sorted scores, we appended valid triple at top
            rank = np.where(sorted_indices_rel.cpu().numpy() == 0)[0][0] + 1
            mr += rank
            mrr += 1.0 / rank

            if rank <= 10:
                h10 += 1
            if rank <= 3:
                h3 +=1
            if rank == 1:
                h1 += 1

    return mrr, h1, h3, h10
