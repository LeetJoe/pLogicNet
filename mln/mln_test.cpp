#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>

#define MAX_STRING 1000
#define MAX_THREADS 100

// 二元函数
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// sigmoid 的反函数
double inv_sigmoid(double x)
{
    return -log(1.0 / x - 1.0);
}

int min(int a, int b)
{
    if (a < b) return a;
    return b;
}

struct Triplet
{
    int h, t, r;
    char type;
    int valid;
    double truth, logit;
    std::vector<int> rule_ids;

    // 构造函数
    Triplet()
    {
        h = -1;
        t = -1;
        r = -1;
        type = -1;
        valid = -1;
        truth = 0;
        logit = 0;
        rule_ids.clear();
    }

    // 析构函数
    ~Triplet()
    {
        rule_ids.clear();
    }
    
    void init()
    {
        truth = 0;
        logit = 0;
        rule_ids.clear();
    }

    // friend 表示“友元”，可以让声明的方法无视封装性约束访问类的私有数据成员
    // < 运算符重载。大小的比较依据是：关系->头->尾
    friend bool operator < (Triplet u, Triplet v)
    {
        if (u.r == v.r)
        {
            if (u.h == v.h) return u.t < v.t;
            return u.h < v.h;
        }
        return u.r < v.r;
    }

    // == 运算符重载。如果 r, h, t 都相等则认为相等
    friend bool operator == (Triplet u, Triplet v)
    {
        if (u.h == v.h && u.t == v.t && u.r == v.r) return true;
        return false;
    }
};

// todo 实体-关系 对？
struct Pair
{
    int e, r;
};

struct Rule
{
    std::vector<int> r_premise;   // premise: 前提
    int r_hypothesis;   // 假设
    std::string type;
    double precision, weight, grad;    // 精度、权重、梯度
    
    Rule()
    {
        precision = 0;
        weight = 0;
        grad = 0;
    }

    // type -> r_hypothesis -> r_premise[0..k]
    friend bool operator < (Rule u, Rule v)
    {
        if (u.type == v.type)
        {
            if (u.r_hypothesis == v.r_hypothesis)
            {
                int min_length = min(int(u.r_premise.size()), int(v.r_premise.size()));
                for (int k = 0; k != min_length; k++)
                {
                    if (u.r_premise[k] != v.r_premise[k])
                    return u.r_premise[k] < v.r_premise[k];
                }
            }
            return u.r_hypothesis < v.r_hypothesis;
        }
        return u.type < v.type;
    }
};


// 全局变量可以直接在函数中访问
char observed_triplet_file[MAX_STRING], probability_file[MAX_STRING], output_rule_file[MAX_STRING], output_prediction_file[MAX_STRING], output_hidden_file[MAX_STRING], save_file[MAX_STRING], load_file[MAX_STRING];
int entity_size = 0, relation_size = 0, triplet_size = 0, observed_triplet_size = 0, hidden_triplet_size = 0, rule_size = 0, iterations = 10, num_threads = 1;
double rule_threshold = 0, triplet_threshold = 1, learning_rate = 0.01;
long long total_count = 0;
std::map<std::string, int> ent2id, rel2id;
std::vector<std::string> id2ent, id2rel; // todo 为什么用 vector?
std::vector<Triplet> triplets;
std::vector<Pair> *h2rt = NULL;
std::set<Rule> candidate_rules;
std::vector<Rule> rules;
std::set<Triplet> observed_triplets, hidden_triplets;
std::map<Triplet, double> triplet2prob;  // 三元组 到 概率 的 map
std::map<Triplet, int> triplet2id;  // todo triplet to id?
std::vector<int> rand_idx; // todo 随机 index ？
sem_t mutex; // 看起来是同步信号量

/* Debug 输出所有 rules 里的 premise relations */
void print_rule(Rule rule)
{
    for (int k = 0; k != int(rule.r_premise.size()); k++) printf("%s ", id2rel[rule.r_premise[k]].c_str());
    printf("-> %s %s\n", id2rel[rule.r_hypothesis].c_str(), rule.type.c_str());
}

/* Debug 输出所有 (h, r, t) 以及 type, valid, truth, logic，还有 rule_ids  */
void print_triplet(Triplet triplet)
{
    int h = triplet.h;
    int t = triplet.t;
    int r = triplet.r;
    printf("%s %s %s\n", id2ent[h].c_str(), id2rel[r].c_str(), id2ent[t].c_str());
    printf("%c %d %lf %lf\n", triplet.type, triplet.valid, triplet.truth, triplet.logit);
    for (int k = 0; k != int(triplet.rule_ids.size()); k++) print_rule(rules[triplet.rule_ids[k]]);
    printf("\n");
    printf("\n");
}

void read_data()
{
    char s_head[MAX_STRING], s_tail[MAX_STRING], s_rel[MAX_STRING];
    int h, t, r;
    Triplet triplet;
    Pair ent_rel_pair;
    std::map<std::string, int>::iterator iter; // todo 这个局部变量 iter 似乎没有用到？
    FILE *fi;
    
    fi = fopen(observed_triplet_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: file of observed triplets not found!\n");
        exit(1);
    }
    while (1)
    {
        if (fscanf(fi, "%s %s %s", s_head, s_rel, s_tail) != 3) break;
        
        if (ent2id.count(s_head) == 0) // 发现一个新的 entity
        {
            // entity_size 用来计数实体数量同时也用于给新发现的实体编号，相当于一个自增id
            ent2id[s_head] = entity_size;
            id2ent.push_back(s_head);
            entity_size += 1;
        }
        
        if (ent2id.count(s_tail) == 0)
        {
            ent2id[s_tail] = entity_size;
            id2ent.push_back(s_tail);
            entity_size += 1;
        }
        
        if (rel2id.count(s_rel) == 0) // 发现一个新的 relation
        {
            rel2id[s_rel] = relation_size;
            id2rel.push_back(s_rel);
            relation_size += 1;
        }

        // id 化以构建 triplet
        h = ent2id[s_head]; t = ent2id[s_tail]; r = rel2id[s_rel];
        // triplet.type 为 o 表示 observed, 相应的 valid 初始值为 1；后面查找的隐三元组其 triplet.type 为 h，valid 初始值为 0.
        triplet.h = h; triplet.t = t; triplet.r = r; triplet.type = 'o'; triplet.valid = 1;
        // 将新发现的 triplet append 进列表中 todo 这里 triplets 是 vector 类型
        triplets.push_back(triplet);
        // 也放进 observed(set 类型) 里，与前面的 triplets 相比，这里应该是去了重的
        observed_triplets.insert(triplet);
    }
    fclose(fi);

    // todo observed triplets 与 triplets 的含义并不相同，这里的变量命名怎么混在一起了？
    observed_triplet_size = int(triplets.size());

    // 这个 h2rt 有全局声明，这里将它实例化了，其内容是 h_id 到 [Pair(r_id, t_id)] 的映射，注意 val 部分是一个 vector
    h2rt = new std::vector<Pair> [entity_size];
    for (int k = 0; k != observed_triplet_size; k++)
    {
        // 这里通过 triplet 来初始化 h2rt，todo 直接在前面遍历输入文件的循环中完成不行吗？
        h = triplets[k].h; r = triplets[k].r; t = triplets[k].t;
        
        ent_rel_pair.e = t;
        ent_rel_pair.r = r;
        h2rt[h].push_back(ent_rel_pair);
    }

    // train.txt 里的数据读取到内存中完毕
    printf("#Entities: %d          \n", entity_size);
    printf("#Relations: %d          \n", relation_size);
    printf("#Observed triplets: %d          \n", observed_triplet_size);
}

// 检查给定的三元组是否是 observed
bool check_observed(Triplet triplet)
{
    // 对 Triplet 结构体进行 < 和 == 的重载应该就是为了完成这里的 count，需要为 set 中的元素定义一套确定大小以及是否相等的规则
    // 对 triplet 进行排序或查重只考虑 h, r, t 不考虑其它如 type, valid 等属性。
    if (observed_triplets.count(triplet) != 0) return true;
    else return false;
}

// 查找“复合规则”: 即存在 x-[r0]->y-[r1]->z 与 x-[r]->z，则将其视为一个“复合规则”，其中 r0、r1 称为 premise，r 称为 hypothesis
// 这里的 rule 跟 TLogic 里的 rule 的组织方式类似，都是一个 h, 然后围绕它将与其相关的 r 和 t 组织在一起（虽然具体形式有所不同）
void search_composition_rules(int h, int r, int t)
{
    int len0, len1, mid, r0, r1;
    Rule rule;
    
    len0 = int(h2rt[h].size()); // 与 h 相关联的 (r, t) pairs 的数量
    for (int k = 0; k != len0; k++)
    {
        // h -> (mid, r0)
        mid = h2rt[h][k].e;
        r0 = h2rt[h][k].r;
        
        len1 = int(h2rt[mid].size()); // 即 h2rt[tail].size()
        for (int i = 0; i != len1; i++)
        {
            // h -[r0]-> mid -[r1]-> e_any
            if (h2rt[mid][i].e != t) continue;

            // h -[r0]-> mid -[r1]-> t 这条路径如果存在。（h, r, t) 是初始的输入三元组，相当于从 h 到 t 找到了一条长度为 2 的路径。
            r1 = h2rt[mid][i].r;
            
            rule.r_premise.clear(); // 因为 rule 变量是一直复用的，要消除之前发现的 rule 的影响
            rule.r_premise.push_back(r0);
            rule.r_premise.push_back(r1); // r0, r1 构成了一条路径，这条路径被放在 r_premise 里
            rule.r_hypothesis = r;
            rule.type = "composition"; // “复合规则”
            candidate_rules.insert(rule);

            // 循环要继续下去，因为可能存在多个 r0+r1 的组合与 r 满足复合规则
        }
    }
}

// 查找“对称规则”：即存在 x-[r]->y 与 y-[r]->x，则将其视为“对称规则”。注意在对称规则里，两个方向的关系是同一个，都是 r (如“朋友”)。
void search_symmetric_rules(int h, int r, int t)
{
    int len;
    Rule rule;
    
    len = int(h2rt[t].size());
    for (int k = 0; k != len; k++)
    {
        if (h2rt[t][k].r != r) continue;
        if (h2rt[t][k].e != h) continue;

        rule.r_premise.clear();
        rule.r_premise.push_back(r);
        rule.r_hypothesis = r;
        rule.type = "symmetric";
        candidate_rules.insert(rule);

        // 这里没有直接 break，可能是觉得对称规则如果存在的话，t-[r]->h 只可能出现
    }
    
}

// 查找“反向规则”：即存在 x-[r0]->y 与 y-[r1]->x，则将其视为“反向规则”，如“老师”和“学生”。也有些孤例，如：老王是小明的“爸爸”同时小明又是老王的“学生”
void search_inverse_rules(int h, int r, int t)
{
    int len, invr;
    Rule rule;
    
    len = int(h2rt[t].size());
    for (int k = 0; k != len; k++)
    {
        if (h2rt[t][k].r == r) continue;
        if (h2rt[t][k].e != h) continue;
        
        invr = h2rt[t][k].r;
        
        rule.r_premise.clear();
        rule.r_premise.push_back(invr);
        rule.r_hypothesis = r;
        rule.type = "inverse";
        candidate_rules.insert(rule);
    }
}

// 查找“子关系规则”：即如果有 x-[r0]->y，则有 x-[r1]->y，如“父亲”与“监护人”通常同时存在，但对成年子女却又不是如此。
void search_subrelation_rules(int h, int r, int t)
{
    int len, subr;
    Rule rule;
    
    len = int(h2rt[h].size());
    for (int k = 0; k != len; k++)
    {
        if (h2rt[h][k].e != t) continue;
        
        subr = h2rt[h][k].r;
        if (subr == r) continue;
        
        rule.r_premise.clear();
        rule.r_premise.push_back(subr);
        rule.r_hypothesis = r;
        rule.type = "subrelation";
        candidate_rules.insert(rule);
    }
}

// 查找候选规则，即遍历所有三元组查找上述四种规则，并进行一些简单的处理（rules、rand_idx）
void search_candidate_rules()
{
    for (int k = 0; k != observed_triplet_size; k++)
    {
        if (k % 100 == 0)
        {
            // 每100轮打印一下处理进度。char 13 是回车符
            printf("Progress: %.3lf%%          %c", (double)k / (double)(observed_triplet_size + 1) * 100, 13);
            fflush(stdout);
        }

        search_composition_rules(triplets[k].h, triplets[k].r, triplets[k].t);
        search_symmetric_rules(triplets[k].h, triplets[k].r, triplets[k].t);
        search_inverse_rules(triplets[k].h, triplets[k].r, triplets[k].t);
        search_subrelation_rules(triplets[k].h, triplets[k].r, triplets[k].t);
    }
    
    std::set<Rule>::iterator iter; // candidate_rules 是 set 型的变量，可以用这种方式进行遍历
    for (iter = candidate_rules.begin(); iter != candidate_rules.end(); iter++)
    rules.push_back(*iter); // rules 是全局的 vector 类型的变量 todo 将 candidate_rules 里的内容直接全部放进去？
    
    rule_size = int(candidate_rules.size());
    candidate_rules.clear(); // 将 candidate_rules 清空
    printf("#Candidate rules: %d          \n", rule_size);

    // 对规则进行 shuffle 形成一个 rand_idx
    for (int k = 0; k != rule_size; k++) rand_idx.push_back(k);
    std::random_shuffle(rand_idx.begin(), rand_idx.end());
}

// 在 triplets 范围内，统计给定的 rule 的有效性(此时的 triplets 里都是 observed)
double precision_composition_rule(Rule rule)
{
    int len, h, mid, t;
    int rp0, rp1, rh;
    double p = 0, q = 0;
    Triplet triplet;
    
    rp0 = rule.r_premise[0];
    rp1 = rule.r_premise[1];
    rh = rule.r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp0) continue;
        
        h = triplets[k].h;
        mid = triplets[k].t;
        
        for (int i = 0; i != int(h2rt[mid].size()); i++)
        {
            if (h2rt[mid][i].r != rp1) continue;
            
            t = h2rt[mid][i].e;
            triplet.h = h; triplet.r = rh; triplet.t = t;
            
            if (check_observed(triplet) == true) p += 1;
            q += 1;
        }
    }
    
    return p / q;
}

// 同上
double precision_symmetric_rule(Rule rule)
{
    int h, t, rp, rh, len;
    double p = 0, q = 0;
    Triplet triplet;
    
    rp = rule.r_premise[0];
    rh = rule.r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (check_observed(triplet) == true) p += 1;
        q += 1;
    }
    
    return p / q;
}

// 同上
double precision_inverse_rule(Rule rule)
{
    int h, t, rp, rh, len;
    double p = 0, q = 0;
    Triplet triplet;
    
    rp = rule.r_premise[0];
    rh = rule.r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (check_observed(triplet) == true) p += 1;
        q += 1;
    }
    
    return p / q;
}

// 同上
double precision_subrelation_rule(Rule rule)
{
    int h, t, rp, rh, len;
    double p = 0, q = 0;
    Triplet triplet;
    
    rp = rule.r_premise[0];
    rh = rule.r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = h; triplet.t = t; triplet.r = rh;
        
        if (check_observed(triplet) == true) p += 1;
        q += 1;
    }
    
    return p / q;
}

// 用于计算规则精度的线程函数
void *compute_rule_precision_thread(void *id)
{
    // 根据 id 与 num_threads 进行数据分配
    int thread = int((long)(id));
    int bg = int(rule_size / num_threads) * thread;
    int ed = int(rule_size / num_threads) * (thread + 1);
    if (thread == num_threads - 1) ed = rule_size;
    
    for (int T = bg; T != ed; T++)
    {
        if (T % 10 == 0)  // 每 10 轮输出一次进度
        {
            total_count += 10;
            printf("Progress: %.3lf%%          %c", (double)total_count / (double)(rule_size + 1) * 100, 13);
            fflush(stdout);
        }

        int k = rand_idx[T];  // 通过 rand_idx 实现对原始三元组的顺序随机化

        if (rules[k].type == "composition") rules[k].precision = precision_composition_rule(rules[k]);
        if (rules[k].type == "symmetric") rules[k].precision = precision_symmetric_rule(rules[k]);
        if (rules[k].type == "inverse") rules[k].precision = precision_inverse_rule(rules[k]);
        if (rules[k].type == "subrelation") rules[k].precision = precision_subrelation_rule(rules[k]);
    }
    
    pthread_exit(NULL);
}

// 使用多线程方式计算规则精度，任务分发到 compute_rule_precision_thread 函数中
void compute_rule_precision()
{
    // 这里按照参数进行线程创建，完成后各 threads 依自己的 id 编号进行数据分配
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    total_count = 0;
    for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, compute_rule_precision_thread, (void *)a);
    for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    free(pt);

    // 只保留那些 precision 达到门限的那些 rules
    std::vector<Rule> rules_copy(rules);
    rules.clear();
    for (int k = 0; k != rule_size; k++)
    {
        // rule_threshold 取自 mln_threshold_of_rule，论文里给的是 0.1 或 0.6，依数据集有所不同
        if (rules_copy[k].precision >= rule_threshold) rules.push_back(rules_copy[k]);
    }
    rules_copy.clear();

    // 更新 rule_size
    rule_size = int(rules.size());
    printf("#Final Rules: %d          \n", rule_size);
}

// 使用“组合规则”查找可能的隐三元组
void search_hidden_with_composition(int id, int thread)
{
    int h, mid, t, len;
    int rp1, rp2, rh;
    Triplet triplet;
    
    rp1 = rules[id].r_premise[0];
    rp2 = rules[id].r_premise[1];
    rh = rules[id].r_hypothesis;

    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp1) continue;
        
        h = triplets[k].h;
        mid = triplets[k].t;
        
        for (int i = 0; i != int(h2rt[mid].size()); i++)
        {
            if (h2rt[mid][i].r != rp2) continue;
            
            t = h2rt[mid][i].e;
            // 找到的 (h, rh, t) 三元组可能原先不存在，也可能原先就存在，还有可能通过其它规则已经发现过了，所以要 check_observed
            triplet.h = h; triplet.t = t; triplet.r = rh;
            
            if (check_observed(triplet) == true) continue;  // 如果已经发现了该隐三元组则跳过下面步骤继续循环
            // 这里的 type 与 valid 有别于前面从文件直接读出的 observed triplets
            triplet.type = 'h'; triplet.valid = 0;
            // todo 同样是使用多线程处理，candidate_rules 那里没有加锁，这里却加了锁。为什么呢？两个变量都是 set 类型。
            // hidden triplets 远远多于 observed，而 rules 又远少于 observed triplets，所以 rules 几乎不可能冲突，故不必加锁？
            sem_wait(&mutex);
            hidden_triplets.insert(triplet);
            sem_post(&mutex);
        }
    }
}

// 使用“对称规则”查找可能的隐三元组
void search_hidden_with_symmetric(int id, int thread)
{
    int h, t, rp, rh, len;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (check_observed(triplet) == true) continue;
        triplet.type = 'h'; triplet.valid = 0;
        sem_wait(&mutex);
        hidden_triplets.insert(triplet);
        sem_post(&mutex);
    }
}

// 使用“逆向规则”查找可能的隐三元组
void search_hidden_with_inverse(int id, int thread)
{
    int h, t, rp, rh, len;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (check_observed(triplet) == true) continue;
        triplet.type = 'h'; triplet.valid = 0;
        sem_wait(&mutex);
        hidden_triplets.insert(triplet);
        sem_post(&mutex);
    }
}

// 使用“子关系规则”查找可能的隐三元组
void search_hidden_with_subrelation(int id, int thread)
{
    int h, t, rp, rh, len;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    len = int(triplets.size());
    for (int k = 0; k != len; k++)
    {
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = h; triplet.t = t; triplet.r = rh;
        
        if (check_observed(triplet) == true) continue;
        triplet.type = 'h'; triplet.valid = 0;
        sem_wait(&mutex);
        hidden_triplets.insert(triplet);
        sem_post(&mutex);
    }
}

// 查找隐三元组的线程 worker 函数
void *search_hidden_triplets_thread(void *id)
{
    // 任务划分
    int thread = int((long)(id));
    int bg = int(rule_size / num_threads) * thread;
    int ed = int(rule_size / num_threads) * (thread + 1);
    if (thread == num_threads - 1) ed = rule_size;
    
    for (int k = bg; k != ed; k++)
    {
        if (k % 10 == 0)
        {
            total_count += 10;
            printf("Progress: %.3lf%%          %c", (double)total_count / (double)(rule_size + 1) * 100, 13);
            fflush(stdout);
        }

        // 进行查找
        if (rules[k].type == "composition") search_hidden_with_composition(k, thread);
        if (rules[k].type == "symmetric") search_hidden_with_symmetric(k, thread);
        if (rules[k].type == "inverse") search_hidden_with_inverse(k, thread);
        if (rules[k].type == "subrelation") search_hidden_with_subrelation(k, thread);
    }
    
    pthread_exit(NULL);
}

// 寻找隐三元组的线程分派与后处理
void search_hidden_triplets()
{
    sem_init(&mutex, 0, 1);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    total_count = 0;
    for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, search_hidden_triplets_thread, (void *)a);
    for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    free(pt);

    // 查找得到的初始的隐三元组数量可能远多于已知三元组，如 demo 中 observed 是48万，发现的 hidden 有240万
    hidden_triplet_size = int(hidden_triplets.size());
    triplet_size = observed_triplet_size + hidden_triplet_size;

    // 将 hidden triplets 合并到 triplets 中
    std::set<Triplet>::iterator iter;
    for (iter = hidden_triplets.begin(); iter != hidden_triplets.end(); iter++) triplets.push_back(*iter);
    printf("#Hidden triplets: %d          \n", hidden_triplet_size);
    printf("#Triplets: %d          \n", triplet_size);
}

// todo neosong
void read_probability_of_hidden_triplets()
{
    if (probability_file[0] == 0)
    {
        Pair ent_rel_pair;
        
        triplet2id.clear();
        for (int k = 0; k != entity_size; k++) h2rt[k].clear();
        
        for (int k = 0; k != triplet_size; k++)
        {
            triplet2id[triplets[k]] = k;
            
            if (triplets[k].type == 'o')
            {
                triplets[k].valid = 1;
                triplets[k].truth = 1;
                ent_rel_pair.e = triplets[k].t;
                ent_rel_pair.r = triplets[k].r;
                h2rt[triplets[k].h].push_back(ent_rel_pair);
            }
            else
            {
                triplets[k].valid = 0;
                triplets[k].truth = 0;
            }
        }
        return;
    }
    
    char s_head[MAX_STRING], s_tail[MAX_STRING], s_rel[MAX_STRING];
    double prob;
    Triplet triplet;
    
    FILE *fi = fopen(probability_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: probability file not found!\n");
        exit(1);
    }
    while (1)
    {
        if (fscanf(fi, "%s %s %s %lf", s_head, s_rel, s_tail, &prob) != 4) break;
        
        if (ent2id.count(s_head) == 0) continue;
        if (ent2id.count(s_tail) == 0) continue;
        if (rel2id.count(s_rel) == 0) continue;
        
        triplet.h = ent2id[s_head];
        triplet.t = ent2id[s_tail];
        triplet.r = rel2id[s_rel];
        
        triplet2prob[triplet] = prob;
    }
    fclose(fi);
    
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].type == 'o')
        {
            triplets[k].truth = 1;
            triplets[k].valid = 1;
            continue;
        }
        
        if (triplet2prob.count(triplets[k]) != 0 && triplet2prob[triplets[k]] >= triplet_threshold)
        {
            triplets[k].truth = triplet2prob[triplets[k]];
            triplets[k].valid = 1;
        }
        else
        {
            triplets[k].truth = triplet2prob[triplets[k]];
            triplets[k].valid = 0;
        }
    }
    
    for (int k = 0; k != entity_size; k++) h2rt[k].clear();
    
    int h, r, t;
    Pair ent_rel_pair;
    for (int k = 0; k != triplet_size; k++)
    {
        triplet2id[triplets[k]] = k;
        
        if (triplets[k].valid == 0) continue;
        
        h = triplets[k].h; r = triplets[k].r; t = triplets[k].t;
        
        ent_rel_pair.e = t;
        ent_rel_pair.r = r;
        h2rt[h].push_back(ent_rel_pair);
    }
}

void link_composition_rule(int id)
{
    int tid, h, mid, t;
    int rp0, rp1, rh;
    Triplet triplet;
    
    rp0 = rules[id].r_premise[0];
    rp1 = rules[id].r_premise[1];
    rh = rules[id].r_hypothesis;
    
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].valid == 0) continue;
        if (triplets[k].r != rp0) continue;
        
        h = triplets[k].h;
        mid = triplets[k].t;
        
        for (int i = 0; i != int(h2rt[mid].size()); i++)
        {
            if (h2rt[mid][i].r != rp1) continue;
            
            t = h2rt[mid][i].e;
            triplet.h = h; triplet.r = rh; triplet.t = t;
            
            if (triplet2id.count(triplet) == 0) continue;
            tid = triplet2id[triplet];
            sem_wait(&mutex);
            triplets[tid].rule_ids.push_back(id);
            sem_post(&mutex);
        }
    }
}

void link_symmetric_rule(int id)
{
    int tid, h, t, rp, rh;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].valid == 0) continue;
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (triplet2id.count(triplet) == 0) continue;
        tid = triplet2id[triplet];
        sem_wait(&mutex);
        triplets[tid].rule_ids.push_back(id);
        sem_post(&mutex);
    }
}

void link_inverse_rule(int id)
{
    int tid, h, t, rp, rh;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].valid == 0) continue;
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = t; triplet.t = h; triplet.r = rh;
        
        if (triplet2id.count(triplet) == 0) continue;
        tid = triplet2id[triplet];
        sem_wait(&mutex);
        triplets[tid].rule_ids.push_back(id);
        sem_post(&mutex);
    }
}

void link_subrelation_rule(int id)
{
    int tid, h, t, rp, rh;
    Triplet triplet;
    
    rp = rules[id].r_premise[0];
    rh = rules[id].r_hypothesis;
    
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].valid == 0) continue;
        if (triplets[k].r != rp) continue;
        
        h = triplets[k].h; t = triplets[k].t;
        triplet.h = h; triplet.t = t; triplet.r = rh;
        
        if (triplet2id.count(triplet) == 0) continue;
        tid = triplet2id[triplet];
        sem_wait(&mutex);
        triplets[tid].rule_ids.push_back(id);
        sem_post(&mutex);
    }
}

void *link_rules_thread(void *id)
{
    int thread = int((long)(id));
    int bg = int(rule_size / num_threads) * thread;
    int ed = int(rule_size / num_threads) * (thread + 1);
    if (thread == num_threads - 1) ed = rule_size;
    
    for (int k = bg; k != ed; k++)
    {
        if (k % 10 == 0)
        {
            total_count += 10;
            printf("Progress: %.3lf%%          %c", (double)total_count / (double)(rule_size + 1) * 100, 13);
            fflush(stdout);
        }

        if (rules[k].type == "composition") link_composition_rule(k);
        if (rules[k].type == "symmetric") link_symmetric_rule(k);
        if (rules[k].type == "inverse") link_inverse_rule(k);
        if (rules[k].type == "subrelation") link_subrelation_rule(k);
    }
    
    pthread_exit(NULL);
}

void link_rules()
{
    sem_init(&mutex, 0, 1);
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    total_count = 0;
    for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, link_rules_thread, (void *)a);
    for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    free(pt);

    printf("Data preprocessing done!          \n");
}

void init_weight()
{
    for (int k = 0; k != rule_size; k++)
    rules[k].weight = (rand() / double(RAND_MAX) - 0.5) / 100;
}

double train_epoch(double lr)
{
    double error = 0, cn = 0;
    
    for (int k = 0; k != rule_size; k++) rules[k].grad = 0;
    
    for (int k = 0; k != triplet_size; k++)
    {
        int len = int(triplets[k].rule_ids.size());
        if (len == 0) continue;
        
        triplets[k].logit = 0;
        for (int i = 0; i != len; i++)
        {
            int rule_id = triplets[k].rule_ids[i];
            triplets[k].logit += rules[rule_id].weight / len;
        }
        
        triplets[k].logit = sigmoid(triplets[k].logit);
        for (int i = 0; i != len; i++)
        {
            int rule_id = triplets[k].rule_ids[i];
            rules[rule_id].grad += (triplets[k].truth - triplets[k].logit) / len;
        }
        
        error += (triplets[k].truth - triplets[k].logit) * (triplets[k].truth - triplets[k].logit);
        cn += 1;
    }
    
    for (int k = 0; k != rule_size; k++) rules[k].weight += lr * rules[k].grad;
    
    return sqrt(error / cn);
}

void output_rules()
{
    if (output_rule_file[0] == 0) return;
    
    FILE *fo = fopen(output_rule_file, "wb");
    for (int k = 0; k != rule_size; k++)
    {
        std::string type = rules[k].type;
        double weight = rules[k].weight;
        
        fprintf(fo, "%s\t%s\t", type.c_str(), id2rel[rules[k].r_hypothesis].c_str());
        for (int i = 0; i != int(rules[k].r_premise.size()); i++)
        fprintf(fo, "%s\t", id2rel[rules[k].r_premise[i]].c_str());
        fprintf(fo, "%lf\n", weight);
    }
    fclose(fo);
}

void output_predictions()
{
    if (output_prediction_file[0] == 0) return;
    
    FILE *fo = fopen(output_prediction_file, "wb");
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].type == 'o') continue;
        
        int h = triplets[k].h;
        int t = triplets[k].t;
        int r = triplets[k].r;
        double prob = triplets[k].logit;
        
        fprintf(fo, "%s\t%s\t%s\t%lf\n", id2ent[h].c_str(), id2rel[r].c_str(), id2ent[t].c_str(), prob);
    }
    fclose(fo);
}

void output_hidden_triplets()
{
    if (output_hidden_file[0] == 0) return;
    
    FILE *fo = fopen(output_hidden_file, "wb");
    for (int k = 0; k != triplet_size; k++)
    {
        if (triplets[k].type == 'o') continue;
        
        int h = triplets[k].h;
        int t = triplets[k].t;
        int r = triplets[k].r;
        
        fprintf(fo, "%s\t%s\t%s\n", id2ent[h].c_str(), id2rel[r].c_str(), id2ent[t].c_str());
    }
    fclose(fo);
}

void save()
{
    if (save_file[0] == 0) return;
    
    FILE *fo = fopen(save_file, "wb");
    
    fprintf(fo, "%d\n", entity_size);
    for (int k = 0; k != entity_size; k++) fprintf(fo, "%d\t%s\n", k, id2ent[k].c_str());
    
    fprintf(fo, "%d\n", relation_size);
    for (int k = 0; k != relation_size; k++) fprintf(fo, "%d\t%s\n", k, id2rel[k].c_str());
    
    fprintf(fo, "%d\n", triplet_size);
    for (int k = 0; k != triplet_size; k++)
    {
        int h = triplets[k].h;
        int r = triplets[k].r;
        int t = triplets[k].t;
        char type = triplets[k].type;
        int valid = triplets[k].valid;
        
        fprintf(fo, "%s\t%s\t%s\t%c\t%d\n", id2ent[h].c_str(), id2rel[r].c_str(), id2ent[t].c_str(), type, valid);
    }
    
    fprintf(fo, "%d\n", rule_size);
    for (int k = 0; k != rule_size; k++)
    {
        std::string type = rules[k].type;
        double weight = rules[k].weight;
        
        fprintf(fo, "%s\t%lf\t%s\t%d\t", type.c_str(), rules[k].precision, id2rel[rules[k].r_hypothesis].c_str(), int(rules[k].r_premise.size()));
        for (int i = 0; i != int(rules[k].r_premise.size()); i++)
        fprintf(fo, "%s\t", id2rel[rules[k].r_premise[i]].c_str());
        fprintf(fo, "%lf\n", weight);
    }
    
    fclose(fo);
}

void load()
{
    if (load_file[0] == 0) return;
    
    FILE *fi = fopen(load_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: loading file not found!\n");
        exit(1);
    }
    
    fscanf(fi, "%d", &entity_size);
    id2ent.clear(); ent2id.clear();
    int eid; char s_ent[MAX_STRING];
    for (int k = 0; k != entity_size; k++)
    {
        fscanf(fi, "%d %s", &eid, s_ent);
        id2ent.push_back(s_ent);
        ent2id[s_ent] = eid;
    }
    
    fscanf(fi, "%d", &relation_size);
    id2rel.clear(); rel2id.clear();
    int rid; char s_rel[MAX_STRING];
    for (int k = 0; k != relation_size; k++)
    {
        fscanf(fi, "%d %s", &rid, s_rel);
        id2rel.push_back(s_rel);
        rel2id[s_rel] = rid;
    }
    
    fscanf(fi, "%d", &triplet_size);
    triplets.clear();
    observed_triplets.clear();
    h2rt = new std::vector<Pair> [entity_size];
    int h, r, t;
    char t_type, s_head[MAX_STRING], s_tail[MAX_STRING];
    int valid;
    Triplet triplet;
    Pair ent_rel_pair;
    observed_triplet_size = 0; hidden_triplet_size = 0;
    for (int k = 0; k != triplet_size; k++)
    {
        fscanf(fi, "%s %s %s %c %d\n", s_head, s_rel, s_tail, &t_type, &valid);
        h = ent2id[s_head]; r = rel2id[s_rel]; t = ent2id[s_tail];
        triplet.h = h; triplet.r = r; triplet.t = t; triplet.type = t_type; triplet.valid = valid;
        triplet.rule_ids.clear();
        triplets.push_back(triplet);
        
        if (t_type == 'o')
        {
            observed_triplets.insert(triplet);
            observed_triplet_size += 1;
        }
        else
        {
            hidden_triplet_size += 1;
        }
        
        if (valid == 0) continue;
        ent_rel_pair.e = t;
        ent_rel_pair.r = r;
        h2rt[h].push_back(ent_rel_pair);
    }
    
    fscanf(fi, "%d", &rule_size);
    rules.clear();
    Rule rule;
    char r_type[MAX_STRING];
    for (int k = 0; k != rule_size; k++)
    {
        int cn;
        fscanf(fi, "%s %lf %s %d", r_type, &rule.precision, s_rel, &cn);
        rule.r_hypothesis = rel2id[s_rel];
        rule.type = r_type;
        rule.r_premise.clear();
        for (int i = 0; i != cn; i++)
        {
            fscanf(fi, "%s", s_rel);
            rule.r_premise.push_back(rel2id[s_rel]);
        }
        fscanf(fi, "%lf", &rule.weight);
        rules.push_back(rule);
    }
    
    fclose(fi);
    
    printf("#Entities: %d          \n", entity_size);
    printf("#Relations: %d          \n", relation_size);
    printf("#Observed triplets: %d          \n", observed_triplet_size);
    printf("#Hidden triplets: %d          \n", hidden_triplet_size);
    printf("#Triplets: %d          n", triplet_size);
    printf("#Rules: %d          \n", rule_size);
}

void train()
{
    if (load_file[0] == 0)
    {
        // Read observed triplets
        read_data();
        // Search for candidate logic rules
        search_candidate_rules();
        // Compute the empirical precision of logic rules and filter out low-precision ones
        compute_rule_precision();
        // Search for hidden triplets with the extracted logic rules
        search_hidden_triplets();
    }
    else
    {
        load();
    }
    
    save();
    output_hidden_triplets();

    if (iterations == 0) return;

    // Read the probability of hidden triplets predicted by KGE models
    read_probability_of_hidden_triplets();
    // Link each triplet to logic rules which can extract the triplet
    link_rules();
    // Initialize the weight of logic rules randomly
    init_weight();
    for (int k = 0; k != iterations; k++)
    {
        double error = train_epoch(learning_rate);
        printf("Iteration: %d %lf          \n", k, error);
    }
    output_rules();
    output_predictions();
}

int ArgPos(char *str, int argc, char **argv)
{
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) 
    {
        if (a == argc - 1) 
        {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv)
{
    int i;
    if (argc == 1)
    {
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-observed <file>\n");
        printf("\t\tFile of observed triplets, one triplet per line, with the format: <h> <r> <t>.\n");
        printf("\t-probability <file>\n");
        printf("\t\tAnnotation of hidden triplets from KGE model, one triplet per line, with the format: <h> <r> <t> <prob>.\n");
        printf("\t-out-rule <file>\n");
        printf("\t\tOutput file of logic rules.\n");
        printf("\t-out-prediction <file>\n");
        printf("\t\tOutput file of predictions on hidden triplets by MLN.\n");
        printf("\t-out-hidden <file>\n");
        printf("\t\tOutput file of discovered hidden triplets.\n");
        printf("\t-save <file>\n");
        printf("\t\tSaving file.\n");
        printf("\t-load <file>\n");
        printf("\t\tLoading file.\n");
        printf("\t-iterations <int>\n");
        printf("\t\tNumber of iterations for training.\n");
        printf("\t-lr <float>\n");
        printf("\t\tLearning rate.\n");
        printf("\t-thresh-rule <float>\n");
        printf("\t\tThreshold for logic rules. Logic rules whose empirical precision is less than the threshold will be filtered out.\n");
        printf("\t-thresh-triplet <float>\n");
        printf("\t\tThreshold for triplets. Hidden triplets whose probability is less than the threshold will be viewed as false ones.\n");
        printf("\t-threads <int>\n");
        printf("\t\tNumber of running threads.\n");
        return 0;
    }
    observed_triplet_file[0] = 0;
    probability_file [0] = 0;
    output_rule_file[0] = 0;
    output_prediction_file[0] = 0;
    output_hidden_file[0] = 0;
    save_file[0] = 0;
    load_file[0] = 0;
    if ((i = ArgPos((char *)"-observed", argc, argv)) > 0) strcpy(observed_triplet_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-probability", argc, argv)) > 0) strcpy(probability_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out-rule", argc, argv)) > 0) strcpy(output_rule_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out-prediction", argc, argv)) > 0) strcpy(output_prediction_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-out-hidden", argc, argv)) > 0) strcpy(output_hidden_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save", argc, argv)) > 0) strcpy(save_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-load", argc, argv)) > 0) strcpy(load_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-iterations", argc, argv)) > 0) iterations = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-lr", argc, argv)) > 0) learning_rate = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-thresh-rule", argc, argv)) > 0) rule_threshold = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-thresh-triplet", argc, argv)) > 0) triplet_threshold = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    train();
    return 0;
}
