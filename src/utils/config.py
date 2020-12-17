import copy

class Config:
    def __init__(
        self,
        cased=False,
        model_ckpt='./artifacts/genia/',
        dataset='genia',
        eval_on_training=False,
        token_emb_dim=200,
        char_emb_dim=60,
        hidden_dim=100,
        total_layers=16,
        batch_size=64,
        max_steps=1e9,
        max_epoches=500,
        dropout=0.45,
        wv_file='./data/glove.6B.200d.txt',
        lm_emb_dim=1024,
        lm_name='dmis-lab/biobert-large-cased-v1.1',
        continue_training=True,
        device=None,
        *args,
        **kargs,
    ):
        self.cased = cased
        self.model_ckpt = model_ckpt
        self.dataset = dataset
        self.eval_on_training = eval_on_training
        self.token_emb_dim = token_emb_dim
        self.char_emb_dim = char_emb_dim
        self.hidden_dim = hidden_dim
        self.total_layers = total_layers
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_epoches = max_epoches
        self.dropout = dropout
        self.wv_file = wv_file
        self.lm_emb_dim = lm_emb_dim
        self.lm_name = lm_name
        self.continue_training = continue_training
        self.device = device
        
    def __call__(self, **kargs):
        obj = copy.copy(self)
        for k, v in kargs.items():
            setattr(obj, k, v)
        return obj
    
    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
