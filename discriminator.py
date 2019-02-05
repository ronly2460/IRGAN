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
            self.item_bias = nn.Embedding(
                self.item_num, 1)
            
            # initialize
            torch.nn.init.uniform_(
                self.user_embeddings.weight, a=initdelta, b=-initdelta)
            torch.nn.init.uniform_(
                self.item_embeddings.weight, a=initdelta, b=-initdelta)
#             torch.nn.init.uniform_(
#                 self.item_bias.weight, a=initdelta, b=-initdelta)
            
    def forward(self, user, item, label):
        pre_logits = (self.user_embeddings(user) \
                              * self.item_embeddings(item)).squeeze().sum(1).view(-1,1) \
                                + self.item_bias(item).view(-1, 1)
        pre_loss =  F.binary_cross_entropy_with_logits(pre_logits, label)
        return pre_loss

    def all_rating(self, user):
        rating = torch.mm(self.user_embeddings(user).view(1, -1),
                                               self.item_embeddings.weight.t()) + self.item_bias
        return all_rating

    def all_logits(self, user):
        logits = (self.user_embeddings(user) \
                              * self.item_embeddings.weight).sum(1) + self.item_bias
        return all_logits

    def get_reward(self, user, item):
        reward_logits = (self.user_embeddings(user) * self.item_embeddings(item)).squeeze().sum(1).view(-1,1) + self.item_bias(item).view(-1, 1)
        reward = 2 * (torch.sigmoid(reward_logits) - 0.5)
        return reward