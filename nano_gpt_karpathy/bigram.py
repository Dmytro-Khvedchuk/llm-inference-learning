import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size: int = 64
block_size: int = 256
max_iters: int = 500
eval_interval: int = 500
learning_rate: float = 3e-4
device = 'mps' if torch.mps.is_available() else 'cpu'

print(f"device: {device}")

eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
weight_decay = 0.1


torch.manual_seed(1337)


with open('LLM/nano_gpt_karpathy/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split) -> tuple[torch.Tensor, torch.Tensor]:
    # get the train split if split==train and else val split
    data = train_data if split == 'train' else val_data
    # create a tensor of batch_size size with random 
    # integers from range (len(data) - block_size)
    ix = torch.randint(len(data) - block_size, (batch_size,)) # (batch_size)
    # then we create a matrix from random data points
    # like 
    # [
    #   [h, e, ..., a] len=block_size
    #   [h, e, ..., a] len=block_size
    #   vert_len=batch_size
    # ]
    x = torch.stack([data[i:i+block_size] for i in ix]) # (batch_size, block_size)
    # same as previous but shift on 1 in the right
    # [
    #   [e, ..., a, b] len=block_size
    #   [e, ..., a, b] len=block_size
    #   vert_len=batch_size
    # ]
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # (batch_size, block_size)

    # load the data to device
    x,y = x.to(device), y.to(device)
    return x, y

# off the gradient for the estimate run
@torch.no_grad()
def estimate_loss():
    out = {}
    # switch model to the eval stage (no dropouts)
    model.eval()
    # for each split in train and val (on the whole dataset)
    for split in ['train', 'val']:
        # initialize the zero vector of the step
        losses = torch.zeros(eval_iters) # [0, 0, ... ,0]. len=200
        # for some k in (1, 200)
        for k in range(eval_iters):
            # get the X, Y split from the whole dataset
            X, Y = get_batch(split)
            # have the logits and loss output from the GPT class
            logits, loss = model(X, Y)
            # write loss for the iteration k
            losses[k] = loss.item()
        # average the loss for all 200 passes for train and val
        out[split] = losses.mean()
    # switch to train mode
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        # inherit the nn.Module
        super().__init__()
        # define the K vector which is just a layer which has a size of n_embd 
        # and the outputs it to a head_size=64, i don't know why we turn the bias off
        self.key = nn.Linear(n_embd, head_size, bias=False)
        # same for Q query
        self.query = nn.Linear(n_embd, head_size, bias=False)
        # same for V value
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # we are moving the tril operation which is constant to a mps device
        # the torch.ones create a matrix with (block_size, block_size) dimenstions of 1
        # then apply the tril, which creates a 0s from the upper triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        

    def forward(self, x):
        # get the x with (B, T, n_embd)
        B, T, C = x.shape
        # make a forward path thorugh the linear layer, and get the
        # (B, T, head_size)
        k = self.key(x)
        # same as k, (B, T, n_embd) -> (B, T, head_size)
        q = self.query(x)

        # weights = q matmul k.transpose(-2, -1) it means that we
        # are multiplying the (B, T, head_size) @ (B, head_size, T)
        # transpose here swaps the last with one before last dimension
        # then the matrix we got we multiply by 1/sqrt(n_embd)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # -> (B, T, T)

        # during the training we have a T fixes, 
        # but during inference we can have it bigger, so we are sizing the 
        # matrix for the current sequence length, then == 0 makes from it the
        # boolean mask, and where True we are filling it with float('-inf')
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # apply softmax over dim=-1 (the last axis, which is the key index).
        # softmax along an axis makes the numbers along that axis sum to 1,
        # so EACH ROW of wei becomes a probability distribution over the keys:
        # every row sums to 1, the columns do NOT. shape stays (B, T, T).
        # (row i = how much query token i attends to each key; these are the
        #  weights used next in wei @ v.)
        wei = F.softmax(wei, dim=-1)

        # same as k and q. (B, T, n_embd) -> (B, T, head_size)
        v = self.value(x)

        # then we have the (B, T, _T_) @ (B, _T_, head_size) -> (B, T, head_size)
        out = wei @ v
        # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        # inherit the nn.Module
        super().__init__()
        # initialize the heads in parallel blocks (single wall) for num_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # define the projection layer (don't know why we have it)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # take the h outout will be the (B, T, head_size), head_size in our case is 64
        # and then concatenate across all of the heads to get (B, T, n_embd) because
        # of the line head_size = n_embd // n_head
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # just make it a pass through one NN layer to mix the heads
        out = self.proj(out)
        # (B, T, n_embd)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        # inherit the nn.Module
        super().__init__()
        # initialize the network which is a sequence of 2 layers with ReLU between them
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        # make a forward pass throught the FFN
        # B, T, n_embd
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        # inherit the nn.Module class
        super().__init__()
        # we define a value of head_size to run them on multi-head
        # each head will have the size of 384 / 6 = 64
        head_size = n_embd // n_head
        # we feed the heads into the MultiHeadAttention block
        # with n_head = 6 heads and head_size = 64 the dimension of one head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # pass the values (B, T, n_embd) for the LayerNorm first and the
        # multi-head attention block to get the forward pass.
        # we keep x + to not have anything on the residual path and gradients 
        # can flow freely. (Need to learn how gradients work and how they flow
        # throught the NNs)
        x = x + self.sa(self.ln1(x))
        # pass this in the FFN with another LayerNorm. Another because in
        # LayerNorm a value that is learning and we want it to be different
        x = x + self.ffwd(self.ln2(x))
        return x


class GenerativePretrainedTransformer(nn.Module):
    def __init__(self):
        # inherit the nn.Module
        super().__init__()
        # initialize the token_embedding_table which is a learnable lookup table
        # in this function we want that our vocab_size tokens were represented as
        # n_embd size vectors. It says what is this token?
        # the dimenstion is (vocab_size, n_embd)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # initialize the position_embedding_table which is a learnable lookup table
        # in this function we have the block_size to have the length of a sequence
        # we are learning in and it is also have the n_embd dimenstions
        # the dimension is (block_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # sequential as I understand that we make the code go block by block, not
        # in parallel, we have the n_layer blocks like n_layer walls
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, targets=None):
        # idx is a tensor with (batch_size, block_size) dimension
        # targets is the same but it is optional
        # for batch = 1 if we for example want to make a decode loop, 
        # the block_size will be the length of our context

        # split the (batch_size, block_size)
        # where batch is the amount of the sequences and block_size is time (length of the context)
        B, T = idx.shape

        # get a token embedding so we put there a (B, T) and get on the output
        # the (B, T, n_embd), so we map every token in (B, T) with a size of n_embd
        # for the communication with the transformer
        tok_emb = self.token_embedding_table(idx)
        # now we do kind of the same but with position table.
        # so we have on the output (T = block_size, n_embd) and it is in the form
        # [
        #   [val], [val] => len = n_embd
        #   [val],
        #   [val],
        #   =>
        #   block_size
        # ]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        # adding the (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd)
        x = tok_emb + pos_emb

        # feeding the x in the blocks (B, T, n_embd)
        x = self.blocks(x)
        # get back the B, T, n_embd
        x = self.ln_f(x)
        # get the logits = the scores for a given token
        # which will return the (B, T, vocab_size)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # turn the logits from (B, T, vocab_size) to (B*T, vocab_size)
            # in other words the sequence and scores for each token for a place
            logits = logits.view(B*T, C)
            # then rurn targets from (B, T) to (B*T)
            targets = targets.view(B*T)
            # compute the cross entropy across all entries
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # we get all batches and the sequence is last block_size tokens (256)
            # if we had a KV cache we would store them, but here we will always recompute the 
            # last 256 tokens to get a new one
            idx_cond = idx[:, -block_size:]
            
            # get the logits, the scores for each token (B, T, vocab_size)
            logits, loss = self(idx_cond)
            # take only the last token of the sequence B, vocab_size
            logits = logits[:, -1, :]
            # softmax to get the probabilities across the vocab_size
            probs = F.softmax(logits, dim=-1)
            # take the 1 random (by distribution) token from the probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            # concatenate the sequence with a new token by the T
            # B, T is idx and we are doing here the (B, T+1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# initialized the model class
model = GenerativePretrainedTransformer()
# loaded the model to device
m = model.to(device)
# initialized the AdamW optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# for each iteration in the range (0, max_iters-1)
for iter in range(max_iters):

    # if iter can be divided cleanly for the eval_interval
    if iter % eval_interval == 0:
        # calculate the loss
        losses = estimate_loss()
        # print the loss function
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # get the data for the train
    # xb as yb are a matrices with (batch_size, block_size)
    xb, yb = get_batch('train')
    # feed the data into the model to get a loss and logits.
    logits, loss = model(xb, yb)
    # wipe old gradients
    optimizer.zero_grad(set_to_none=True)
    # compute te gradients
    loss.backward()
    # apply the update
    optimizer.step()

# have the context of [[0]]
context = torch.zeros((1, 1), dtype = torch.long, device=device)

# print the sequence of 500 tokens and the B, T input will be [[0]]
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
