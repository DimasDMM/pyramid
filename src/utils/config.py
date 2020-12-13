import copy

class Config:
    def __init__(
        self,
        cased=False,
        model_ckpt='./artifacts/genia/',
        dataset='genia',
        token_emb_dim=100,
        char_emb_dim=100,
        hidden_dim=100,
        total_layers=16,
        batch_size=64,
        max_steps=1e9,
        max_epoches=100,
        dropout=0.4,
        wv_file='./data/glove.6B.100d.txt',
        freeze_wv=True,
        lm_emb_dim=768,
        lm_name='dmis-lab/biobert-v1.1',
        device=None,
        *args,
        **kargs,
    ):
        self.cased = cased
        self.model_ckpt = model_ckpt
        self.dataset = dataset
        self.token_emb_dim = token_emb_dim
        self.char_emb_dim = char_emb_dim
        self.hidden_dim = hidden_dim
        self.total_layers = total_layers
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.max_epoches = max_epoches
        self.dropout = dropout
        self.wv_file = wv_file
        self.freeze_wv = freeze_wv
        self.lm_emb_dim = lm_emb_dim
        self.lm_name = lm_name
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
