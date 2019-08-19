from pytorch_transformers import *
import torch
from matplotlib import pyplot as pp
import numpy as np

tokenizer = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103');
model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')

# Tokenize input
text = "I am beginning to think this model is strange and [MASK]"
tokenized_text = tokenizer.tokenize(text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

with torch.no_grad():
    outputs = model(tokens_tensor)
    prediction_scores, mems = outputs[:2]

# Visualize MEM states
ndx_L = 17
n_M = model.config.mem_len
n_M_hist = 20
ndx_M = n_M-n_M_hist
pp.figure(); pp.plot(outputs[1][ndx_L-1][ndx_M,0,:].data); pp.title('Transformer-XL MEM [layer:{}, ndx:{}/{}]'.format(ndx_L,ndx_M,n_M)); pp.show()
pp.figure(); pp.imshow(outputs[1][ndx_L-1][n_M-tokens_tensor.shape[-1]:,0,:].data); pp.title('Transformer-XL MEM [layer:{}, ndx:{}/{}]'.format(ndx_L,ndx_M,n_M)); pp.show()

# Last 5 words of input text
print(tokenizer.decode(tokens_tensor[:,-5:].squeeze()))

# Display what model 'thinks' are the most likely top-k words to substitute for masked tokens
top_k = 20
for n,tok in enumerate(tokenized_text):
    print('[{}] '.format(tok),tokenizer.decode(torch.argsort(prediction_scores[:, n].squeeze(0)).tolist()[-1:-top_k - 1:-1]))

## NOTE:  these results seem inferior to bert branch results