class Generator(nn.Module):
    
    def __init__(self, item_num, user_num, emb_dim,
                           lamda, param=None, initdelta=0.05, lr=0.05):
        super(Generator, self).__init__()
        self.item_num = item_num
        self.user_num = user_num
        self.emb_dim = emb_dim
        self.lamda = lamda
        self.param = param
        self.initdelta = initdelta
        self.lr = lr
        self.g_params = []
        
        if self.param is None:
            self.user_embeddings = nn.Embedding(
                self.user_num, self.emb_dim)
            self.item_embeddings = nn.Embedding(
                self.item_num, self.emb_dim)
            self.item_bias = torch.zeros(self.item_num)
            
            # initialize
            torch.nn.init.uniform_(
                self.user_embeddings.weight, a=initdelta, b=-initdelta)
            torch.nn.init.uniform_(
                self.item_embeddings.weight, a=initdelta, b=-initdelta)
        
    def forward(self, user, item, reward):
        softmax_score = F.softmax(self.all_logits(user).view(1, -1), -1)
        i_prob = torch.gather(softmax_score.view(-1), 0, item).clamp(min=1e-8)
        loss = - torch.mean(torch.log(i_prob) * reward) # \
#                    + self.lamda * (F.normalize(self.user_embeddings(user), p=2, dim=1) \
#                                              + F.normalize(self.item_embeddings(item), p=2, dim=1) \
#                                              + F.normalize(self.item_bias(item), p=2, dim=1))
        return loss

    def all_rating(self, user):
        rating = torch.mm(self.user_embeddings(user).view(1, -1),
                                               self.item_embeddings.weight.t()) + self.item_bias
        return rating
    
    def all_logits(self, user):
        logits = (self.user_embeddings(user) \
                              * self.item_embeddings.weight).sum(1) + self.item_bias
        return logits