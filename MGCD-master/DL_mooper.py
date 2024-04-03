import torch
from torch.utils.data import DataLoader
import json
import params
import random
import numpy as np
import dgl


def my_collate(batch):
    input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
    for log in batch:
        knowledge_codes = []
        if params.type == "equal":
            import Grain_mooper
            knowledge_codes2 = Grain_mooper.equal2(log['exer_id'])
            knowledge_codes = knowledge_codes2+log['knowledge_code']
        else:
            knowledge_codes = log['knowledge_code']
        if knowledge_codes == []:
            knowledge_emb = [1.0] * params.kn
        else:
            knowledge_emb = [0.] * params.kn
            for knowledge_code in knowledge_codes:
                knowledge_emb[knowledge_code] = 1.0
        y = log['score']
        input_stu_ids.append(log['user_id'])
        input_exer_ids.append(log['exer_id'])
        input_knowledge_embs.append(knowledge_emb)
        # input_knowledge_embs.append(knowledge_codes)
        ys.append(y)

    return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.LongTensor(input_knowledge_embs), torch.Tensor(ys)

def CD_DL():
    with open(params.src) as i_f:
        src_dataset = json.load(i_f)
    with open(params.tgt) as i_f:
        tgt_dataset = json.load(i_f)
    src_DL = DataLoader(dataset=src_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=my_collate)
    tgt_DL = DataLoader(dataset=tgt_dataset, batch_size=params.batch_size, shuffle=True, collate_fn=my_collate)
    return src_DL, tgt_DL

# def build_graph():
#     src = []
#     dst = []
#     graph_list = params.graph
#     for i in graph_list:
#         with open(i, 'r') as f: #e k
#             for line in f.readlines():
#                 line = line.replace('\n', '').split('\t')
#                 src.append(int(line[0]))
#                 dst.append(int(line[1]))
#     src1 = src+dst
#     dst1 = dst+src
#     g = dgl.graph((src1, dst1))
#
#     return g
#
# def build_graph1():
#     src = []
#     dst = []
#     with open(params.graph, 'r') as f: #e k
#         for line in f.readlines():
#             line = line.replace('\n', '').split('\t')
#             src.append(int(line[0]))
#             dst.append(int(line[1]))
#     g = dgl.graph((src, dst))
#     g = dgl.add_self_loop(g)
#     return g
