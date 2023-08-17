import sys
import os
import datetime
from utils_test import augment_triplet, evaluate

dataset = 'data/FB15k'
path = './record'

iterations = 2

kge_model = 'TransE'
kge_batch = 1024
kge_neg = 256
kge_dim = 100
kge_gamma = 24
kge_alpha = 1
kge_lr = 0.001
kge_iters = 10000
kge_tbatch = 16
kge_reg = 0.0
kge_topk = 100

# 对不同的图嵌入模型和不同的数据集，使用不同的参数设置（所以又是一个调参调出来的最优）
if kge_model == 'RotatE':
    if dataset.split('/')[-1] == 'FB15k':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 1024, 256, 1000, 24.0, 1.0, 0.0001, 150000, 16
    if dataset.split('/')[-1] == 'FB15k-237':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 1024, 256, 1000, 9.0, 1.0, 0.00005, 100000, 16
    if dataset.split('/')[-1] == 'wn18':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 512, 1024, 500, 12.0, 0.5, 0.0001, 80000, 8
    if dataset.split('/')[-1] == 'wn18rr':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 512, 1024, 500, 6.0, 0.5, 0.00005, 80000, 8

if kge_model == 'TransE':
    if dataset.split('/')[-1] == 'FB15k':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 1024, 256, 1000, 24.0, 1.0, 0.0001, 150000, 16
    if dataset.split('/')[-1] == 'FB15k-237':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 1024, 256, 1000, 9.0, 1.0, 0.00005, 100000, 16
    if dataset.split('/')[-1] == 'wn18':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 512, 1024, 500, 12.0, 0.5, 0.0001, 80000, 8
    if dataset.split('/')[-1] == 'wn18rr':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch = 512, 1024, 500, 6.0, 0.5, 0.00005, 80000, 8

if kge_model == 'DistMult':
    if dataset.split('/')[-1] == 'FB15k':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 1024, 256, 2000, 500.0, 1.0, 0.001, 150000, 16, 0.000002
    if dataset.split('/')[-1] == 'FB15k-237':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 1024, 256, 2000, 200.0, 1.0, 0.001, 100000, 16, 0.00001
    if dataset.split('/')[-1] == 'wn18':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 200.0, 1.0, 0.001, 80000, 8, 0.00001
    if dataset.split('/')[-1] == 'wn18rr':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 512, 1024, 1000, 200.0, 1.0, 0.002, 80000, 8, 0.000005

if kge_model == 'ComplEx':
    if dataset.split('/')[-1] == 'FB15k':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 1024, 256, 1000, 500.0, 1.0, 0.001, 150000, 16, 0.000002
    if dataset.split('/')[-1] == 'FB15k-237':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 1024, 256, 1000, 200.0, 1.0, 0.001, 100000, 16, 0.00001
    if dataset.split('/')[-1] == 'wn18':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 512, 1024, 500, 200.0, 1.0, 0.001, 80000, 8, 0.00001
    if dataset.split('/')[-1] == 'wn18rr':
        kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, kge_reg = 512, 1024, 500, 200.0, 1.0, 0.002, 80000, 8, 0.000005


# 这几个参数只跟数据集有关，跟图嵌入模型无关
if dataset.split('/')[-1] == 'FB15k':
    mln_threshold_of_rule = 0.1
    mln_threshold_of_triplet = 0.7
    weight = 0.5
if dataset.split('/')[-1] == 'FB15k-237':
    mln_threshold_of_rule = 0.6
    mln_threshold_of_triplet = 0.7
    weight = 0.5
if dataset.split('/')[-1] == 'wn18':
    mln_threshold_of_rule = 0.1
    mln_threshold_of_triplet = 0.5
    weight = 100
if dataset.split('/')[-1] == 'wn18rr':
    mln_threshold_of_rule = 0.1
    mln_threshold_of_triplet = 0.5
    weight = 100

mln_iters = 1000
mln_lr = 0.0001
mln_threads = 8

# ------------------------------------------

# 如果目录不存在刚创建
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# 根据模型构建生成图嵌入的 shell 命令，其中 model 是模型的名称，而 workspace_path 是包含了迭代轮次的路径
def cmd_kge(workspace_path, model):
    if model == 'RotatE':
        return 'bash ./kge/kge_test.sh train {} {} 0 {} {} {} {} {} {} {} {} {} {} -de'.format(model, dataset, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk)
    if model == 'TransE':
        return 'bash ./kge/kge_test.sh train {} {} 0 {} {} {} {} {} {} {} {} {} {}'.format(model, dataset, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk)
    if model == 'DistMult':
        return 'bash ./kge/kge_test.sh train {} {} 0 {} {} {} {} {} {} {} {} {} {} -r {}'.format(model, dataset, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk, kge_reg)
    if model == 'ComplEx':
        return 'bash ./kge/kge_test.sh train {} {} 0 {} {} {} {} {} {} {} {} {} {} -de -dr -r {}'.format(model, dataset, kge_batch, kge_neg, kge_dim, kge_gamma, kge_alpha, kge_lr, kge_iters, kge_tbatch, workspace_path, kge_topk, kge_reg)

# 使用 mln 程序生成隐状态和 MLN 或使用保存的 MLN 生成预测和规则
def cmd_mln(main_path, workspace_path=None, preprocessing=False):
    if preprocessing == True:
        return './mln/mln -observed {}/train.txt -out-hidden {}/hidden.txt -save {}/mln_saved.txt -thresh-rule {} -iterations 0 -threads {}'.format(main_path, main_path, main_path, mln_threshold_of_rule, mln_threads)
    else:
        return './mln/mln -load {}/mln_saved.txt -probability {}/annotation.txt -out-prediction {}/pred_mln.txt -out-rule {}/rule.txt -thresh-triplet 1 -iterations {} -lr {} -threads {}'.format(main_path, workspace_path, workspace_path, workspace_path, mln_iters, mln_lr, mln_threads)

# 保存命令参数组合，应该是用来跟运行结果相比较，找到最好的一组结果好做表格
def save_cmd(save_path):
    with open(save_path, 'w') as fo:
        fo.write('dataset: {}\n'.format(dataset))
        fo.write('iterations: {}\n'.format(iterations))
        fo.write('kge_model: {}\n'.format(kge_model))
        fo.write('kge_batch: {}\n'.format(kge_batch))
        fo.write('kge_neg: {}\n'.format(kge_neg))
        fo.write('kge_dim: {}\n'.format(kge_dim))
        fo.write('kge_gamma: {}\n'.format(kge_gamma))
        fo.write('kge_alpha: {}\n'.format(kge_alpha))
        fo.write('kge_lr: {}\n'.format(kge_lr))
        fo.write('kge_iters: {}\n'.format(kge_iters))
        fo.write('kge_tbatch: {}\n'.format(kge_tbatch))
        fo.write('kge_reg: {}\n'.format(kge_reg))
        fo.write('mln_threshold_of_rule: {}\n'.format(mln_threshold_of_rule))
        fo.write('mln_threshold_of_triplet: {}\n'.format(mln_threshold_of_triplet))
        fo.write('mln_iters: {}\n'.format(mln_iters))
        fo.write('mln_lr: {}\n'.format(mln_lr))
        fo.write('mln_threads: {}\n'.format(mln_threads))
        fo.write('weight: {}\n'.format(weight))

# 每次运行都单独记录结果，包括所执行的命令的参数组合
'''
time = str(datetime.datetime.now()).replace(' ', '_')
path = path + '/' + time
ensure_dir(path)
save_cmd('{}/cmd.txt'.format(path))
'''
path = './record/2023-08-16_16:56:00.007952'

# ------------------------------------------

# 每次也把相应的数据集的 train.txt 文件也复制一份到新建的目录里
# cp data/FB15k/train.txt ./record/2023-08-16_16:56:00.007952/train.txt
str_cmd = 'cp {}/train.txt {}/train.txt'.format(dataset, path)
print(str_cmd)
# os.system(str_cmd)

# train_augmented.txt 初始使用 train.txt
# cp data/FB15k/train.txt ./record/2023-08-16_16:56:00.007952/train_augmented.txt
str_cmd = 'cp {}/train.txt {}/train_augmented.txt'.format(dataset, path)
print(str_cmd)
# os.system(str_cmd)

# 基于 train.txt 生成 hidden.txt 和 mln_saved.txt
# ./mln/mln -observed ./record/2023-08-16_xxx/train.txt -out-hidden ./record/2023-08-16_xxx/hidden.txt
#   -save ./record/2023-08-16_16:56:00.007952/mln_saved.txt -thresh-rule 0.1 -iterations 0 -threads 8
str_cmd = cmd_mln(path, preprocessing=True)
print(str_cmd)
# os.system(str_cmd)


print('(......start iterations......)')
for k in range(iterations):
    print('(......iterations {}/{}......)'.format(str(k+1), str(iterations)))

    # 如果执行多次迭代，每一次迭代的结果也都单独记录
    workspace_path = path + '/' + str(k)
    ensure_dir(workspace_path)

    # cp ./record/2023-08-16_xxx/train_augmented.txt ./record/2023-08-16_xxx/0/train_kge.txt
    str_cmd = 'cp {}/train_augmented.txt {}/train_kge.txt'.format(path, workspace_path)
    print(str_cmd)
    # os.system(str_cmd)

    # cp ./record/2023-08-16_xxx/hidden.txt ./record/2023-08-16_xxx/0/hidden.txt
    str_cmd = 'cp {}/hidden.txt {}/hidden.txt'.format(path, workspace_path)
    print(str_cmd)
    # os.system(str_cmd)

    # ./kge/kge_test.sh train TransE data/FB15k 0 1024 256 1000 24.0 1.0 0.0001 150000 16 ./record/2023-08-16_xxx/0 100
    str_cmd = cmd_kge(workspace_path, kge_model)
    print(str_cmd)
    # os.system(str_cmd)

    # ./mln/mln -load ./record/2023-08-16_xxx/mln_saved.txt -probability ./record/2023-08-16_xxx/0/annotation.txt
    #   -out-prediction ./record/2023-08-16_xxx/0/pred_mln.txt -out-rule ./record/2023-08-16_xxx/0/rule.txt
    #   -thresh-triplet 1 -iterations 1000 -lr 0.0001 -threads 8
    str_cmd = cmd_mln(path, workspace_path, preprocessing=False)
    print(str_cmd)
    # os.system(str_cmd)

    # 把 pred_mln.txt 里概率大于 threshold 的三元组作为新的三元组和 train.txt 里的数据一起组成 train_augmented.txt 里的三元组数据
    print('(......augment triplet......)')
    # augment_triplet('{}/pred_mln.txt'.format(workspace_path), '{}/train.txt'.format(path), '{}/train_augmented.txt'.format(workspace_path), mln_threshold_of_triplet)

    # 将更新过的 train_augmented.txt 文件复制回 path 将原始的文件覆盖，下次迭代会用它来生成 kge
    # cp ./record/2023-08-16_xxx/0/train_augmented.txt ./record/2023-08-16_xxx/train_augmented.txt
    str_cmd = 'cp {}/train_augmented.txt {}/train_augmented.txt'.format(workspace_path, path)
    print(str_cmd)
    # os.system(str_cmd)

    # 最终评估：计算 MRR, Hit@K 等
    print('(......evaluate......)')
    # evaluate('{}/pred_mln.txt'.format(workspace_path), '{}/pred_kge.txt'.format(workspace_path), '{}/result_kge_mln.txt'.format(workspace_path), weight)

print('(......done......)')
