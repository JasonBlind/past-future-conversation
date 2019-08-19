from pytorch_transformers import *
import torch
from matplotlib import pyplot as pp
import numpy as np

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased', cache_dir='pretrained')
model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased', cache_dir='pretrained')

# Tokenize input
text = "This is some filler text to increase the length of the input so that performance may improve. The multiverse is a <mask> "
input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # We will predict the masked token
mask_token = tokenizer.encode('<mask>')
ndx_masked_tokens = (input_ids==mask_token[0]).nonzero()[:,1].tolist()
perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
perm_mask[:, :, ndx_masked_tokens] = 1.0  # Previous tokens don't see last token
target_mapping = torch.zeros((1, len(ndx_masked_tokens), input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
target_mapping[0, 0, ndx_masked_tokens] = 1.0
#target_mapping[0, 1, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
#target_mapping[0, :3, [-1,-2,-3]] = 1.0 # Our first (and only) prediction will be the last token of the sequence (the masked token)

with torch.no_grad():
    outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    prediction_scores = outputs[0]  # ***prediction_scores** Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

print(tokenizer.decode(input_ids[0,:].tolist()))
print()

top_k = 20
for n,ndx in enumerate(ndx_masked_tokens):
    print('[{}] '.format(ndx),tokenizer.decode(torch.argsort(prediction_scores[:, n].squeeze(0)).tolist()[-1:-top_k - 1:-1]))

# NOTE: I haven't figured out how to properly use perm_mask and target_mapping yet