import tensorflow as tf
from dis_model import DIS
from gen_model import GEN
import cPickle
import numpy as np
import utils as ut
import multiprocessing

EMB_DIM = 5
USER_NUM = 943
ITEM_NUM = 1683
BATCH_SIZE = 16
INIT_DELTA = 0.05

all_items = set(range(ITEM_NUM))
workdir = "ml-100k/"
DIS_TRAIN_FILE = workdir + "dis-train.txt"

def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def simple_test_one_user(x):
    # import pdb; pdb.set_trace()
    rating = x[0]
    u = x[1]

    test_items = list(all_items - set(user_pos_train[u]))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])

def simple_test(model):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    test_users = list(user_pos_test.keys())
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = generator.module.all_rating(user_batch)
        user_batch_rating = user_batch_rating.detach_().cpu().numpy()

        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret

# positiveな要素だけを引っ張ってくる
user_pos_train = {}
with open(workdir + 'movielens-100k-train.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

# testでnegativeな要素だけを引っ張ってくる                
user_pos_test = {}
with open(workdir + 'movielens-100k-test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

all_users = user_pos_train.keys()

def generate_for_d(model, filename):
    data = []
    for user in user_pos_train:
        pos = user_pos_train[user]

        rating = model.all_rating(user)
        rating = np.array(rating[0]) / 0.2  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)

        neg = np.random.choice(np.arange(ITEM_NUM), size=len(pos), p=prob)
        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))
        
generator = Generator(
    ITEM_NUM, USER_NUM,EMB_DIM, lamda=0.0 / BATCH_SIZE,
    param=None, initdelta=INIT_DELTA)

discriminator = Discriminator(
    ITEM_NUM, USER_NUM,EMB_DIM, lamda=0.0 / BATCH_SIZE,
    param=None, initdelta=INIT_DELTA)

g_optimizer = torch.optim.SGD(
    generator.parameters(), lr=0.001, momentum=0.9)

g_optimizer = torch.optim.SGD(
    discriminator.parameters(), lr=0.001, momentum=0.9)                    

for epoch in range(15):
    if epoch >= 0:
        for d_epoch in range(100):
            if d_epoch % 5 == 0:
                generate_for_d(generator, DIS_TRAIN_FILE)
                train_size = ut.file_len(DIS_TRAIN_FILE)
            index = 1
            while True:
                if index > train_size:
                    break
                if index + BATCH_SIZE <= train_size + 1:
                    users, items, labels = ut.get_batch_data(
                        DIS_TRAIN_FILE, index, BATCH_SIZE)
                else:
                    users, items, labels = ut.get_batch_data(
                        DIS_TRAIN_FILE, index, train_size - index + 1)

                loss_d = discriminator(users, items, labels)
                d_optimizer.zero_grad()
                loss_d.backward()
                d_optimizer.step()

        for g_epoch in range(50):
            for user in user_pos_train:
                sample_lambda = 0.2
                pos = user_pos_train[user]

                rating = generator.all_logits(user)
                exp_rating = np.exp(rating)
                prob = exp_rating / np.sum(exp_rating)

                pn = (1 - sample_lambda) + prob
                pn[pos] += sample_lambda * 1.0 / len(pos)

                sample = np.random.choice(np.range(ITEM_NUM), 2*len(pos), p=pn)
                reward = discriminator.get_reward(user, sample)
                reward = reward * prob[sample] / pn[sample]

#                     users = # userを複数に
                loss_g = generator(user, sample, "label", reward)
                g_optimizer.zero_grad()
                loss_g.backward()
                g_optimizer.step()