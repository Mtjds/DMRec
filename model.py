#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author:
# @Date  : 2021/11/1 16:16
# @Desc  :
import os.path
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss, InfoNCE


class GraphEncoder(nn.Module):
    def __init__(self, layers, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)
        self.eps = 0.2

    def forward(self, x, edge_index):

        result = [x]
        all_embeddings = []
        all_embeddings_cl = x
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            # x = self.dropout(x)
            # random_noise = torch.rand_like(x).cuda()
            # x += torch.sign(x) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(x)
            if i == len(self.gnn_layers) - 1:
                all_embeddings_cl = x

            x = F.normalize(x, dim=-1)
            result.append(x / (i + 1))
            # result.append(x)

        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)
        return result, final_embeddings, all_embeddings_cl


class CascadeGraphEncoder(nn.Module):
    def __init__(self, layers, hidden_dim, dropout):
        super(CascadeGraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)
        self.eps = 0.2

    def forward(self, x, edge_index):
        result = [x]
        all_embeddings = []
        all_embeddings_cl = x
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
            x = self.dropout(x)
            # random_noise = torch.rand_like(x).cuda()
            # x += torch.sign(x) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(x)
            if i == len(self.gnn_layers) - 1:
                all_embeddings_cl = x

            x = F.normalize(x, dim=-1)
            result.append(x / (i + 1))
            # result.append(x)

        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        result = torch.stack(result, dim=0)
        result = torch.sum(result, dim=0)
        return result, final_embeddings, all_embeddings_cl
        # for i in range(len(self.gnn_layers)):
        #     x = self.gnn_layers[i](x=x, edge_index=edge_index)
        #     # x = self.dropout(x)
        # return x,x,x


class Mutual_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, dim_qk, dim_v):
        super(Mutual_Attention, self).__init__()
        self.q = nn.Linear(input_dim, dim_qk, bias=False)
        self.k = nn.Linear(input_dim, dim_qk, bias=False)
        self.v = nn.Linear(dim_v, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_qk)

    def forward(self, q_token, k_token, v_token):
        Q = self.q(q_token)  # Q: batch_size * seq_len * dim_k
        K = self.k(k_token)  # K: batch_size * seq_len * dim_k
        V = self.v(v_token)  # V: batch_size * seq_len * dim_v

        # # Q * K.T() # batch_size * seq_len * seq_len
        # att = nn.Softmax(dim=-1)(torch.matmul(Q, K.transpose(-1, -2)) * self._norm_fact)
        #
        # # Q * K.T() * V # batch_size * seq_len * dim_v
        # att = torch.matmul(att, V)

        att = nn.Softmax(dim=-1)(torch.matmul(q_token, k_token.transpose(-1, -2)) * self._norm_fact)
        # 对每个 (4, 4) 的子张量的每一列求平均，得到形状为 (41739, 4) 的张量
        column_means = att.mean(dim=1)
        # 将所有 (41739, 4) 向量逐项求平均，得到形状为 (4,) 的向量
        sum_of_means = column_means.mean(dim=0)
        att = torch.matmul(att, v_token)

        return att,sum_of_means


class DMRec(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(DMRec, self).__init__()

        self.device = args.device
        self.layer = args.layer
        self.layers = args.layers
        self.node_dropout = args.node_dropout
        self.message_dropout = nn.Dropout(p=args.message_dropout)
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.all_edge_index = dataset.all_edge_index
        self.item_behaviour_degree = dataset.item_behaviour_degree.to(self.device)
        self.user_behaviour_degree = dataset.user_behaviour_degree.to(self.device)
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.Graph_encoder = nn.ModuleDict({
            behavior: GraphEncoder(self.layers[index], self.embedding_size, self.node_dropout) for index, behavior in enumerate(self.behaviors)
        })

        self.cascade_graph_encoder = nn.ModuleDict({
            behavior: CascadeGraphEncoder(self.layers[index], self.embedding_size, self.node_dropout) for index, behavior in enumerate(self.behaviors)
        })

        self.global_graph_encoder = GraphEncoder(self.layer, self.embedding_size, self.node_dropout)
        self.W = nn.Parameter(torch.ones(len(self.behaviors)))

        self.dim_qk = args.dim_qk
        self.dim_v = args.dim_v
        self.attention = Mutual_Attention(self.embedding_size, self.dim_qk, self.dim_v)

        self.reg_weight = args.reg_weight
        self.layers = args.layers
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model
        self.message_dropout = nn.Dropout(p=args.message_dropout)

        self.cl_rate = 0.0001
        self.temp = 0.15

        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        self.apply(self._init_weights)

        self.linear_layer = nn.Linear(self.embedding_size, self.embedding_size)
        self.WW = nn.Parameter(torch.randn(self.embedding_size, self.embedding_size))


        self._load_model()





    def _init_weights(self, module):

        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self, total_embeddings,user_behaviors_embeddings,item_behaviors_embeddings):
        """
        gcn propagate in each behavior
        """
        all_user_embeddings, all_item_embeddings = [], []
        final_embeddings, all_embeddings_cl = [], []
        index=0
        for behavior in self.behaviors:
            indices = self.edge_index[behavior].to(self.device)

            temp_embeddings=torch.cat([user_behaviors_embeddings[index], item_behaviors_embeddings[index]], dim=0)

            total_embeddings=F.normalize(total_embeddings, dim=-1)+temp_embeddings

            behavior_embeddings, embeddings, cl_embeddings = self.Graph_encoder[behavior](total_embeddings, indices)
            final_embeddings.append(embeddings)
            all_embeddings_cl.append(cl_embeddings)
            # behavior_embeddings = F.normalize(behavior_embeddings, dim=-1)
            # all_embeddings.append(behavior_embeddings + total_embeddings)

            user_embedding, item_embedding = torch.split(behavior_embeddings, [self.n_users + 1, self.n_items + 1])
            all_user_embeddings.append(user_embedding)
            all_item_embeddings.append(item_embedding)
            index=index+1

        # target_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        # target_user_embeddings = torch.sum(target_user_embeddings, dim=1)
        # all_user_embeddings[-1] = target_user_embeddings

        all_user_embeddings = torch.stack(all_user_embeddings, dim=1)
        all_item_embeddings = torch.stack(all_item_embeddings, dim=1)
        return all_user_embeddings, all_item_embeddings, final_embeddings, all_embeddings_cl

    def gcn(self, total_embeddings, indices):
        # behavior_embeddings=total_embeddings
        all_embeddings = {}
        final_embeddings, all_embeddings_cl = [], []
        for behavior in self.behaviors:


            layer_embeddings = total_embeddings
            indices = self.edge_index[behavior].to(self.device)
            layer_embeddings,embeddings, cl_embeddings= self.cascade_graph_encoder[behavior](layer_embeddings, indices)
            final_embeddings.append(embeddings)
            all_embeddings_cl.append(cl_embeddings)

            layer_embeddings = F.normalize(layer_embeddings, dim=-1)
            # layer_embeddings = torch.relu(torch.matmul(self.linear_layer(layer_embeddings), self.WW))
            total_embeddings = layer_embeddings + total_embeddings



            all_embeddings[behavior] = total_embeddings

        # total_embeddings,_,__ = self.global_graph_encoder(total_embeddings, indices.to(self.device))

        # behavior_embeddings = F.normalize(behavior_embeddings, dim=-1)

        # return total_embeddings + behavior_embeddings

        return all_embeddings['buy'],final_embeddings,all_embeddings_cl
        # in_embeddings = {}
        # out_embeddings = {}
        # all_embeddings = {}
        #
        # total_embeddings = 0
        # embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        # # embeddings = self.gcn(embeddings, self.all_edge_index)
        # # embeddings=self.Graph_encoder['click'](embeddings, self.edge_index['click'].to(self.device))
        # init_embeddings = embeddings
        # for behavior in self.behaviors:
        #     indices = self.edge_index[behavior].to(self.device)
        #     layer_embeddings ,_,_ = self.Graph_encoder[behavior](embeddings, indices)
        #
        #     layer_embeddings= layer_embeddings + F.normalize(init_embeddings, dim=-1)
        #
        #     in_embeddings[behavior] = layer_embeddings
        #
        #     layer_embeddings ,_,_= self.Graph_encoder[behavior](layer_embeddings, indices)
        #
        #     out_embeddings[behavior] = layer_embeddings
        #
        #     # layer_embeddings = F.normalize(layer_embeddings, dim=-1)
        #     init_embeddings = torch.relu(torch.matmul(self.linear_layer(layer_embeddings), self.WW))
        #
        #     total_embeddings += layer_embeddings
        #     all_embeddings[behavior] = total_embeddings
        # user_all_embedding, item_all_embedding = [], []
        # for key, value in all_embeddings.items():
        #     user_embedding, item_embedding = torch.split(all_embeddings[key], [self.n_users + 1, self.n_items + 1])
        #     user_all_embedding.append(user_embedding)
        #     item_all_embedding.append(item_embedding)
        #
        # user_all_embedding = torch.stack(user_all_embedding, dim=1)
        # item_all_embedding = torch.stack(item_all_embedding, dim=1)
        #
        # # return embeddings, in_embeddings, out_embeddings, all_embeddings, user_all_embedding, item_all_embedding
        # return all_embeddings['buy']

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = idx[0]
        i_idx = idx[1]
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    def forward(self, batch_data):
        self.storage_user_embeddings = None
        self.storage_item_embeddings = None

        first_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        all_embeddings = []
        all_u_embeddings = []
        all_i_embeddings = []
        for i in range(0, len(self.behaviors)):
            temp_embeddings, _, __ = self.Graph_encoder[self.behaviors[i]](first_embeddings,
                                                                           self.edge_index[self.behaviors[i]].to(
                                                                               self.device))
            temp_user_embedding, temp_item_embedding = torch.split(temp_embeddings,
                                                                   [self.n_users + 1, self.n_items + 1])
            all_u_embeddings.append(temp_user_embedding)
            all_i_embeddings.append(temp_item_embedding)
        cascade_embeddings,final_embeddings1, all_embeddings_cl1= self.gcn(first_embeddings, self.all_edge_index)
        first_embeddings, _, __ = self.Graph_encoder[self.behaviors[0]](first_embeddings,self.edge_index[self.behaviors[0]].to(self.device))

        cascade_user_embedding, cascade_item_embedding = torch.split(cascade_embeddings, [self.n_users + 1, self.n_items + 1])
        # user_embedding, item_embedding = torch.split(first_embeddings, [self.n_users + 1, self.n_items + 1])

        all_user_embeddings, all_item_embeddings, final_embeddings2, all_embeddings_cl2 = self.gcn_propagate(first_embeddings,all_u_embeddings,all_i_embeddings)

        fusion_user,importance=self.attention(all_user_embeddings, all_user_embeddings, all_user_embeddings)
        # target_user_embeddings=all_user_embeddings


        # all_user_embeddings = all_user_embeddings + torch.stack(all_u_embeddings, dim=1)
        all_user_embeddings = fusion_user+cascade_user_embedding.unsqueeze(1)
        fusion_user=torch.sum(fusion_user, dim=1)
        cascade_user=torch.sum(cascade_user_embedding.unsqueeze(1), dim=1)
        # all_user_embeddings = all_user_embeddings + user_embedding.unsqueeze(1)

        weight = self.item_behaviour_degree * self.W
        weight = weight / (torch.sum(weight, dim=1).unsqueeze(-1) + 1e-8)

        fusion_item=all_item_embeddings * weight.unsqueeze(-1)
        fusion_item=torch.sum(fusion_item, dim=1)
        all_item_embeddings = all_item_embeddings * weight.unsqueeze(-1)
        target_item_embeddings = all_item_embeddings

        # all_i_embeddings = torch.stack(all_i_embeddings, dim=1)
        # a=all_i_embeddings* weight.unsqueeze(-1)

        # all_item_embeddings = torch.sum(all_item_embeddings, dim=1) + torch.sum(a, dim=1)
        all_item_embeddings = fusion_item+cascade_item_embedding
        # all_item_embeddings = torch.sum(all_item_embeddings, dim=1) + item_embedding

        total_loss = 0

        click_user_embeddings=0
        click_item_embeddings = 0



        for i in range(len(self.behaviors)):
            data = batch_data[:, i]
            users = data[:, 0].long()
            items = data[:, 1:].long()

            user_embedding1, item_embedding1 = torch.split(final_embeddings1[i], [self.n_users + 1, self.n_items + 1])
            cl_user_embedding1, cl_item_embedding1 = torch.split(all_embeddings_cl1[i],[self.n_users + 1, self.n_items + 1])
            user_embedding2, item_embedding2 = torch.split(final_embeddings2[i], [self.n_users + 1, self.n_items + 1])
            cl_user_embedding2, cl_item_embedding2 = torch.split(all_embeddings_cl2[i],[self.n_users + 1, self.n_items + 1])

            single_cl_loss1 = 0.0001 * self.cal_cl_loss([users, items[0]],torch.cat([user_embedding1, F.normalize(cl_user_embedding1, dim=-1)], dim=0) , user_embedding1,torch.cat([item_embedding1, F.normalize(cl_item_embedding1, dim=-1)], dim=0), item_embedding1)
            single_cl_loss2 = 0.0003 * self.cal_cl_loss([users, items[0]], torch.cat([cl_user_embedding2, F.normalize(user_embedding2, dim=-1)], dim=0), cl_user_embedding2,torch.cat([cl_item_embedding2, F.normalize(item_embedding2, dim=-1)], dim=0), cl_item_embedding2)
            multi_cl_loss = 0.0
            # 使用线性分配的权重的嵌入做多行为对比的目标行为
            multi_cl_loss=0.0001 * self.cal_cl_loss([users, items[0]], cl_user_embedding2,torch.cat([cl_user_embedding2, F.normalize(fusion_user, dim=-1)], dim=0) ,cl_item_embedding2,torch.cat([cl_item_embedding2, F.normalize(fusion_item, dim=-1)], dim=0))
            # multi_cl_loss = 0.0001 * self.cal_cl_loss([users, items[0]],
            #                                           cl_user_embedding2,
            #                                           cl_user_embedding2+ F.normalize(fusion_user, dim=-1),
            #                                           cl_item_embedding2,
            #                                           cl_item_embedding2+F.normalize(fusion_item, dim=-1))
            # target_user_embeddings,target_item_embeddings=torch.split(all_embeddings_cl2[i],[self.n_users + 1, self.n_items + 1])
            # view_cl_loss=0.0001 * self.cal_cl_loss([users, items[0]], fusion_user, cascade_user,fusion_item, cascade_item_embedding)
            #     使用浏览信息做多行为对比的目标行为
            # if i==0:
            #     click_user_embeddings = cl_user_embedding2
            #     click_item_embeddings = cl_item_embedding2
            # elif i>0:
            #     multi_cl_loss=0.0001 * self.cal_cl_loss([users, items[0]], click_user_embeddings, cl_user_embedding2,click_item_embeddings, cl_item_embedding2)




            view_cl_loss=0.00011  * self.cal_cl_loss([users, items[0]], fusion_user, cascade_user,fusion_item, cascade_item_embedding)



            user_feature = all_user_embeddings[:, i][users.view(-1, 1)]
            item_feature = all_item_embeddings[items]
            # user_feature, item_feature = self.message_dropout(user_feature), self.message_dropo
            # ut(item_feature)
            scores = torch.sum(user_feature * item_feature, dim=2)
            # total_loss += self.bpr_loss(scores[:, 0], scores[:, 1])
            # total_loss += self.bpr_loss(scores[:, 0], scores[:, 1]) + single_cl_loss
            total_loss += self.bpr_loss(scores[:, 0], scores[:, 1]) + single_cl_loss1+single_cl_loss2+multi_cl_loss+view_cl_loss
        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight,
                                                                  self.item_embedding.weight)

        return total_loss

    def full_predict(self, users):
        if self.storage_user_embeddings is None:
            all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

            all_u_embeddings = []
            all_i_embeddings = []
            for i in range(0, len(self.behaviors)):
                temp_embeddings, _, __ = self.Graph_encoder[self.behaviors[i]](all_embeddings,
                                                                               self.edge_index[self.behaviors[0]].to(
                                                                                   self.device))
                temp_user_embedding, temp_item_embedding = torch.split(temp_embeddings,
                                                                       [self.n_users + 1, self.n_items + 1])
                all_u_embeddings.append(temp_user_embedding)
                all_i_embeddings.append(temp_item_embedding)

            cascade_embeddings,_,_ = self.gcn(all_embeddings, self.all_edge_index)
            cascade_user_embedding, cascade_item_embedding = torch.split(cascade_embeddings,[self.n_users + 1, self.n_items + 1])
            all_embeddings, _, __ = self.Graph_encoder[self.behaviors[0]](all_embeddings,self.edge_index[self.behaviors[0]].to(self.device))
            # all_embeddings = self.gcn(all_embeddings, self.all_edge_index)

            user_embedding, item_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])

            all_user_embeddings, all_item_embeddings, _, _ = self.gcn_propagate(all_embeddings,all_u_embeddings,all_i_embeddings)

            target_embeddings = all_user_embeddings[:, -1].unsqueeze(1)
            target_embeddings,_ = self.attention(target_embeddings, all_user_embeddings, all_user_embeddings)
            # self.storage_user_embeddings = target_embeddings.squeeze() + cascade_user_embedding
            self.storage_user_embeddings = target_embeddings.squeeze()+user_embedding

            weight = self.item_behaviour_degree * self.W
            weight = weight / (torch.sum(weight, dim=1).unsqueeze(-1) + 1e-8)
            all_item_embeddings = all_item_embeddings * weight.unsqueeze(-1)
            # self.storage_item_embeddings = torch.sum(all_item_embeddings, dim=1) + cascade_item_embedding
            self.storage_item_embeddings = torch.sum(all_item_embeddings, dim=1)+item_embedding

        user_emb = self.storage_user_embeddings[users.long()]
        scores = torch.matmul(user_emb, self.storage_item_embeddings.transpose(0, 1))

        return scores