import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

def Generator():
    def __init__(self, item_num, user_num, 
                            emb_dim, lamda, param=None, initdelta=0.05, lr=0.05):
        self.item_num = item_num
        self.user_num = user_num
        self.emb_dim = emb_dim
        self.lamda = lamda
        self.param = param
        self.initdelta = initdelta
        self.lr = lr
        self.g_params = []
        
        if self.param is None:
            self.user_embeddings = nn.Embedding() # (self.user_num, self.emb_dim)
            self.item_embeddings = nn.Embedding() # (self.item_num, self.emb_dim)
            self.item_bias = torch.zeros([], dtype=) # (self.item_num)
        else:
            # parameterの読み込み
        
        def forward(self, user, item, label, reward):
            i_prob = torch.gather(F.softmax(self.all_logits(user).view(1, -1), -1), item)
            loss = - (torch.log(i_prob) * reward).sum(1)
                       + self.lamda * (F.normalize(self.user_embeddings(user), p=2, dim=1)
                                                 + F.normalize(self.item_embeddings(item), p=2, dim=1)
                                                 + F.normalize(self.item_bias(item), p=2, dim=1)
            return loss
            
        def all_logits(self, user):
            u_embedding = self.user_embeddings(user)
            all_logits = (self.user_embeddings(user) \
                                  * self.item_embeddings).sum(1) + self.item_bias
            return all_logits