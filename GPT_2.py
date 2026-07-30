from dataclasses import dataclass
import torch
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head ==0    


        self.n_embd = config.n_embd 
        #note that we don't necessarily need n_embd, but importing the GPT-2 weights needs compatiblility
        self.n_head = config.n_head

        #K,Q,V are multi-head
        self.c_attn = nn.Linear(config.n_embd, config.n_embd*3) #biase in this layer can be 'sort of useful'
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 #flag
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size,config.block_size)
                                                .view(1,1,config.block_size,config.block_size)))
        #the triangular matrix is just a tag that instructs where to put '-inf' later
        #additional dimensions created for batching

    def forward(self, x): #x: (B, n_embd)
        B,T,C = x.size()

        qkv = self.c_attn(x) #(B, T, 3*n_embd)
        Q,K,V = qkv.split(self.n_embd, dim=2) #(B, T, C) 

        #(B,T,C) -> (B,T, nh*hs) -> (B,T,nh,hs) -(^T)-> (B,nh,T,hs)
        Q = Q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        K = K.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        V = V.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        att = (Q @ K.transpose(2,3)) * (K.size(-1)**-0.5) # (B,nh,T,hs) @ (B,nh,hs,T) = (B,nh,T,T)
        att = att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf')) 
        att = F.softmax(att, dim=-1)
        y = att @ V # (B,nh,T,T) @ (B,nh,T,hs) = (B,nh,T,hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)

        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        #remember the FFN should have a high dimension to fully blend non-linear relationships
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        #We use approximated GELU because historically the exact version is slow in tensorflow
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        #One layernorm BEFORE each computation layer
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        #observing from the backward pass direction, this is a map-reduce process
        #attn->reduce (compress the inter-token info), mlp->map (independent for each element)


    def forward(self, x):
        #Different from original Transformer, GPT-2 maintains a clean residual stream:
        #We don't do layernorm after residual addition;
        #Instead, we keep one clean (additive) residual stream,
        #which can be represented by: x+f(x)+g(x)+h(x)+...
        x = x + self.attn(self.ln_1(x))   
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int=1024
    vocab_size: int=50257
    n_embd: int=768
    n_layer: int=12
    n_head: int=12

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict( dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), #text embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), #positional embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #transformer layers
            ln_f = nn.LayerNorm(config.n_embd) # the final layernorm
        ) )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #Final linear head

        #weight tying scheme; covering 30% of parameters in training
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                #check 'Kaiming Init'/'batchnorm scaling'/'transformer normalizaiton'
                #('MLP'+'Attn')*n_layer additive residual streams in total, recorded in GPT2 paper
                std *= (2*self.config.n_layer)**-0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            #**in GPT2, positional embeddings have 0.01 deviation 
            nn.init.normal_(module.weight, mean=0.0, std=0.02) 
            #Statistically this should be normalized as well, but we would like embeddings to be a "info base/prior representation",
            #since they represent the model's sensitivity towards input tokens.
            #And this error does not grow with n_layer (like later layers), so we keep the embedding at a 0.02 std
            #In "x0 + dx1+ dx2+...", embeddings are the x0.


    def forward(self, idx, targets=None):
        B, T = idx.size() #always pass in a (B,T)
        assert T<=self.config.block_size, f"cannot forward sequence of length {T}, since it exceeds the max context length"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #generate positional index
        pos_emb = self.transformer.wpe(pos) #embeds position
        tok_emb = self.transformer.wte(idx) #embeds tokens
        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            #reshape logits and targets to (B*T, vocab_size) and (B*T,) respectively for loss computation
            logits = logits.view(B*T, -1) #logits of different possibilities 
            targets = targets.view(B*T) #ground truth
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    @classmethod #analogous to 'constructor' in C
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict() #the variable table, check notebook for details
        sd_keys = sd.keys() #checklist from our side
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()  #state_dict() generates a dict {key(tensor name): value (tensor REFERENCE)}

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape  #reversed shape are equal when one tensor is another's transpose
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())   #special copy enables copying over references
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

class DataLoaderLite:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Total token length is: {len(tokens)}")
        print(f"1 Epoch contains {len(tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T

        #getting the next buffer zone
        buf = self.tokens[self.current_position : self.current_position + B*T +1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        self.current_position += B*T

        if self.current_position + B*T +1 > len(self.tokens): #note that the range above is [,), so use '>'
            self.current_position = 0
        return x,y
#---------------------------------------------------------------------------------------------------------------------------
import time

num_return_sequences = 5
max_length = 30
my_device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print("using device:", my_device) 

torch.random.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.random.manual_seed(1337)

#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
#model.eval() #using evaluation mode, shutoff special layers like dropout, do minor optimization (maybe)
model.to(my_device)

train_loader = DataLoaderLite(B=16, T=1024)

#Check GPT_1.ipynb for AdamW.
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.tiem()

    x, y = train_loader.next_batch()
    #note that the DataLoaderLite class is a CPU-based class, 
    #so we need to ship tensors to GPU explicityly
    x = x.to(my_device)
    y = y.to(my_device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step() #perform a single optimization step

    #wait for GPU to finish the current epoch, so that timing is not just counting the CPU ("GPU queue assigning") time
    torch.cuda.synchronize() 
    t1 = time.time()
    dt = (t1-t0)*1000
    print(f"step {i}: loss {loss.item()}, dt={dt:.2f}ms")
    #notice that .item() is able to carry var back to CPU and print.

import sys; sys.exit(0)





tokens = enc.encode("Hello, I'm a language model.")
tokens = torch.tensor(tokens, dtype = torch.long)
#unsqueeze: insert dimension at shape[x]; repeat: repeat tensor block by x rows, y times per row
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) 
x = tokens.to(my_device) #now x (B, T) -> (5, 8)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) <= max_length:
    with torch.no_grad():
        logits = model(x) #(B, T, vocab_size)
        logits = logits[:,-1,:] #only taking the last on time dimension (newly generated)
        probs = F.softmax(logits, dim = -1) #softmax across the vocab dim, (B, vocab_size)
        #cut every possibilities after top 50 to 0, to avoid bizzare results
        topk_probs, topk_ind = torch.topk(probs, 50, dim=-1) #(B, 50) topk_ind records token ID (i_th of 50257)
        ix = torch.multinomial(topk_probs, 1) #(B, 1) now sampling makes the next token deterministic
        xcol = torch.gather(topk_ind, -1, ix) # collect the ix_th element of each row (which is dim=-1)
        x = torch.cat((x,xcol), dim=-1)

for i in range(num_return_sequences):
    out_tokens = x[i, :max_length].tolist()
    decoded = enc.decode(out_tokens)
    print(">",decoded)


