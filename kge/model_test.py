#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader_test import TestDataset

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        # todo 把 gamma 转化成了一个 Parameter ？
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )

        # todo embedding range ?
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )

        # todo 需要加倍直接传参不就可以了，单独弄个 double_* 参数控制何必呢？
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        # 初始化实体嵌入的矩阵
        # nn.Parameter() 这个函数，正如之前发现的那样，它用来存放一些 tensor 型的数据，能在 parameters() 遍历中找到它，所以它通常起辅助作用，
        # 用来存放那些在训练中要用到的非模型本身的 tensors 如一些参数或者中间状态等，而且如果需要将它存储到 pth 文件中的时候，也很方便找到它。
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))

        # uniform_ 是指 uniform distribution, 使用均匀分布来初始化填充输入的 tensor, a 是它的下限, b 是它的上限。
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        # 初始化 relation 的嵌入矩阵
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))

        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        # todo 如果是 pRotatE, 需要另外定义一个参数 self.modules
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        # 原来这个 double_xxx 是跟着模型走的，对于某些模型，必须把维度加倍。todo 虽然还不清楚为什么
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        # 同上
        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).

        head-batch 里，positive 和 negative 的 head 和 relation 是相同的；
        tail-batch 里，positive 和 negative 的 relation 和 tail 是相同的；
        '''

        if mode == 'single':
            # 取的是 batch 和 negative sample
            batch_size, negative_sample_size = sample.size(0), 1

            # 下面这三句应该都是形成的是一个 m * n 的矩阵，m 是数据量，n 是嵌入表示的维度
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)

        #
        elif mode == 'head-batch':
            tail_part, head_part = sample   # todo 这是什么表达方式？
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)   # todo -1 is head ?
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]   # todo 1 is relation ?
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]   # todo 2 is tail ?
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]   # todo 0 is head ?
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]  # todo 1 is relation ?
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)  # todo -1 is tail ?
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)

        # 按 model name 确定 score 的计算函数
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        # norm 这里是指“范数”，p 为数字时，表示的是向量范数 p-norm. 当 p=1 时，表示对 score 的所有元素的绝对值求和。
        # todo 注意这里的 dim，跟其它地方出现的 dim 一样，不要直接把它用行、列这种直观的方式来认识，容易出现混乱，要用列出 shape 的方式来理解。
        #   如 a.shape=[3, 4], 当 dim=0 时，其含义是在第 0 维上进行求合，进行4次数字个数为3的求合，得到一个4(相当于把shape=[3,4]里的第0维3给消除了)维向量；
        #   若 dim=1 ，就是进行3次数字个数为4的求合，得到一个3(相当于把shape=[3, 4]里的第1维4给消除了)维向量。对更高维的处理更直观，比如一个 shape=[3,4,5]
        #   的矩阵按 dim=0 求范数，将得到一个 shape=[4,5] 的结果；dim=1时，结果 shape=[3, 5]; dim=2 时结果 shape=[3,4]
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        # 对使用 Dataset 构成的 Iterator 进行遍历, 每次得到的都是一个 batch
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling: # todo 这里传参为 True
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries: # 这步不会执行, 给出的数据集里不涉及对 countries 的处理
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            # --------------------------------------------------
            # Comments by Meng:
            # Here we slightly modify the codes to save the intermediate prediction results of KGE models, so that we can combine the predictions from KGE and MLN to improve the results.
            # --------------------------------------------------
            
            predictions = []

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    # 这里的 positive_sample, negative_sample 都是尺寸为 batch 的批量结果
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        # Save prediction results
                        prediction = positive_sample.data.cpu().numpy().tolist()

                        batch_size = positive_sample.size(0)

                        # 直接将 model 作为方法使用, 调用的实际上是 forward() 方法
                        score = torch.sigmoid(model((positive_sample, negative_sample), mode))
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        # torch.sort 就是按指定的维度进行排序, 返回的 valsort 是排序结果, 而 argsort 则是 valsort 原来的下标.
                        # todo 这里 dim=1 表示, valsort 里各一级子项位置不变, 子项的内容(即二级子项)按降序排序; argsort 里各子项位
                        #  置也不变, 子项的元素表示 valsort 各二级子项的原下标.
                        valsort, argsort = torch.sort(score, dim = 1, descending=True)

                        # 取 postive sample 中真实存在的 head/tail
                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            # argsort[i, :] 表示 score 的第 i 项(是一个列表)接降序排序后的下标排列
                            # nonzero() 返回输入非 0 元素的坐标, 根据输入维度不同, 返回的维度也相应变化
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # For each test triplet, save the ranked list (h, r, [ts]) and ([hs], r, t)
                            if mode == 'head-batch':
                                prediction[i].append('h')
                                prediction[i].append(ranking.item() + 1)
                                # zip(a, b) 方法的作用是把 a, b 两个 list 按照对应位置连接成一个 (a_i, b_i) 的 tuple, 构成一个新的 list.
                                ls = zip(argsort[i, 0:args.topk].data.cpu().numpy().tolist(), valsort[i, 0:args.topk].data.cpu().numpy().tolist())
                                prediction[i].append(ls)
                            elif mode == 'tail-batch':
                                prediction[i].append('t')
                                prediction[i].append(ranking.item() + 1)
                                ls = zip(argsort[i, 0:args.topk].data.cpu().numpy().tolist(), valsort[i, 0:args.topk].data.cpu().numpy().tolist())
                                prediction[i].append(ls)

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MR': float(ranking),
                                'MRR': 1.0/ranking,
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        # list 的 + 就是打平拼接.
                        predictions += prediction

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics, predictions

    # --------------------------------------------------
    # Comments by Meng:
    # Here we add a new function, which will predict the probability of each hidden triplet being true.
    # The results will be used by MLN.
    # --------------------------------------------------

    @staticmethod
    def infer_step(model, infer_triples, args):
        batch_size = args.batch_size
        scores = []
        model.eval()
        for k in range(0, len(infer_triples), batch_size):
            bg = k
            ed = min(k + batch_size, len(infer_triples))
            batch = infer_triples[bg:ed]
            batch = torch.LongTensor(batch)
            if args.cuda:
                batch = batch.cuda()
            score = torch.sigmoid(model(batch)).squeeze(1)
            scores += score.data.cpu().numpy().tolist()

        return scores
