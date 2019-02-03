import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class Discriminator(nn.Module):
    
    def __init__(self, item_num, user_num, emb_dim,
                           lamda, param=None, initdelta=0.05):
        super(Discriminator, self).__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.emb_dim = emb_dim
        self.lamda = lamda
        self.param = param
        self.initdelta = initdelta
        self.d_params = []
        
        if self.param is None:
            self.user_embeddings = nn.Embedding(
                self.user_num, self.emb_dim)
            self.item_embeddings = nn.Embedding(
                self.item_num, self.emb_dim)
            self.item_bias = torch.zeros(self.item_num) # (self.item_num)
            
            # initialize
            torch.nn.init.uniform_(
                self.user_embeddings.weight, a=initdelta, b=-initdelta)
            torch.nn.init.uniform_(
                self.item_embeddings.weight, a=initdelta, b=-initdelta)
        else:
            # parameterの読み込み
            print("aaa")
            
        def forward(self, user, item, label):
            pre_logits = (self.user_embeddings(user) \
                                   * self.item_embeddings(item)).sum(1) + self.item_bias(item)
            pre_loss =  F.binary_cross_entropy_with_logits(pre_logits, label)
            return pre_loss

        def all_rating(self, user):
            all_rating = torch.mm(self.user_embeddings(user),
                                                   self.item_embeddings.t()) + self.item_bias
            return all_rating
            
        def all_logits(self, user):
            u_embedding = self.user_embeddings(user)
            all_logits = (self.user_embeddings(user) \
                                  * self.item_embeddings).sum(1) + self.item_bias
            return all_logits
        
        def get_reward(self, user, item):
            reward_logits = (self.user_embeddings(user) \
                                         * self.item_embeddings(item).t()).sum(1) + self.item_bias(item)
            reward = 2 * (torch.nn.Sigmoid(reward_logits) - 0.5)
            return reward