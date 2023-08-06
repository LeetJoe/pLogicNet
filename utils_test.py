import sys
import os

# This function computes the probability of a triplet being true based on the MLN outputs.
# 如果 hrt2p 里对应的(h,r,t)的概率 p 存在，则返回 p，如果 p 不存在，返回 0.5；另外，如果 h==t，则 p 不小于 0.5 时才返回 p，不然返回 -100；
def mln_triplet_prob(h, r, t, hrt2p):
    # KGE algorithms tend to predict triplets like (e, r, e), which are less likely in practice.
    # Therefore, we give a penalty to such triplets, which yields some improvement.
    if h == t:
        if hrt2p.get((h, r, t), 0) < 0.5:
            return -100
        return hrt2p[(h, r, t)]
    else:
        if (h, r, t) in hrt2p:
            return hrt2p[(h, r, t)]
        return 0.5

# This function reads the outputs from MLN and KGE to do evaluation.
# Here, the parameter weight controls the relative weights of both models.
def evaluate(mln_pred_file, kge_pred_file, output_file, weight):
    # MR：Mean Rank; MRR: Mean Reciprocal Rank; Hit@K(H@K)
    # MR, Mean Rank, 均值秩，是指每个观测样本中所有观测值的秩的平均值；
    # MRR, Mean Reciprocal Rank, 中文称为“平均倒数秩”。在排序结果中，如果第 n 个结果正确，那么取秩为 1/n 显然 n 越小 1/n 越大。
    # Hit@K, 是指在所有预测结果中，“在概率最大的前 K 个之前”命中的意思。
    hit1 = 0
    hit3 = 0
    hit10 = 0
    mr = 0
    mrr = 0
    cn = 0

    hrt2p = dict()  # 将 mln 预测组织为 {(h,r,t): p} 形式的字典
    with open(mln_pred_file, 'r') as fi:
        for line in fi:
            h, r, t, p = line.strip().split('\t')[0:4]
            hrt2p[(h, r, t)] = float(p)

    with open(kge_pred_file, 'r') as fi:
        while True:
            # 所以 kge_pred_file 里的内容是按一行实际、一行预测的方式成对存放的
            truth = fi.readline()
            preds = fi.readline()

            if (not truth) or (not preds): # 到文件结尾了或者遇上了单行（意外）
                break

            # truth 看起来是一个六元组
            truth = truth.strip().split()
            # preds 是用空格分隔的 a1:b1 这样的元，后面会拆分成 [[a1, b1],[a2, b2]] 这样的形式
            preds = preds.strip().split()

            # mode 看起来有 t 和 h 两个值，分别是 head 和 tail 的缩写，表示 preds 里的信息是对 head 还是 tail 的预测
            h, r, t, mode, original_ranking = truth[0:5]
            original_ranking = int(original_ranking) # 将 string 转换为 int

            if mode == 'h': # 如果是对 head 的预测
                preds = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds]

                for k in range(len(preds)):
                    e = preds[k][0]
                    # preds[k][1] 最初就有值，这里又在里面累加了其它值
                    preds[k][1] += mln_triplet_prob(e, r, t, hrt2p) * weight

                # 将 preds 按新的 preds[:][1] 进行倒序排序（最上面的是概率最大的）
                preds = sorted(preds, key=lambda x:x[1], reverse=True)
                ranking = -1
                # 找到这组预测命中的位置，即 ranking(从 1 开始计数)
                for k in range(len(preds)):
                    e = preds[k][0]
                    if e == h:
                        ranking = k + 1
                        break
                # 如果都不命中，将 ranking 重置为初始时从 truth 里得到的 ranking
                if ranking == -1:
                    ranking = original_ranking

            if mode == 't': # 与前面一样，这里是对 tail 的预测
                preds = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds]

                for k in range(len(preds)):
                    e = preds[k][0]
                    preds[k][1] += mln_triplet_prob(h, r, e, hrt2p) * weight

                preds = sorted(preds, key=lambda x:x[1], reverse=True)
                ranking = -1
                for k in range(len(preds)):
                    e = preds[k][0]
                    if e == t:
                        ranking = k + 1
                        break
                if ranking == -1:
                    ranking = original_ranking

            # 累计
            if ranking <= 1:
                hit1 += 1
            if ranking <=3:
                hit3 += 1
            if ranking <= 10:
                hit10 += 1
            mr += ranking
            mrr += 1.0 / ranking
            cn += 1

    # 取平均（计算比率）
    mr /= cn
    mrr /= cn
    hit1 /= cn
    hit3 /= cn
    hit10 /= cn

    # 输出结果
    print('MR: ', mr)
    print('MRR: ', mrr)
    print('Hit@1: ', hit1)
    print('Hit@3: ', hit3)
    print('Hit@10: ', hit10)

    # 记录到文件里
    with open(output_file, 'w') as fo:
        fo.write('MR: {}\n'.format(mr))
        fo.write('MRR: {}\n'.format(mrr))
        fo.write('Hit@1: {}\n'.format(hit1))
        fo.write('Hit@3: {}\n'.format(hit3))
        fo.write('Hit@10: {}\n'.format(hit10))

def augment_triplet(pred_file, trip_file, out_file, threshold):
    with open(pred_file, 'r') as fi:
        data = []  # 是一个四元组组成的列表
        for line in fi:
            l = line.strip().split()
            data += [(l[0], l[1], l[2], float(l[3]))]

    with open(trip_file, 'r') as fi:
        trip = set() # 是一个三元组组成的集合
        for line in fi:
            l = line.strip().split()
            trip.add((l[0], l[1], l[2]))

        for tp in data: # 按 threshold 从 data 里面取三元组
            if tp[3] < threshold:
                continue
            trip.add((tp[0], tp[1], tp[2]))

    # 将 trip 里的三元组保存到 out_file 中
    with open(out_file, 'w') as fo:
        for h, r, t in trip:
            fo.write('{}\t{}\t{}\n'.format(h, r, t))
