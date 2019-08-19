from pytorch_transformers import *
import torch
from matplotlib import pyplot as pp
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased');
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Tokenize input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():
    outputs = model(tokens_tensor, masked_lm_labels=tokens_tensor, token_type_ids=segments_tensors)
    encoded_layers = outputs[0]
    prediction_scores = outputs[1]

# Display what model 'thinks' are the most likely top-k words to substitute for masked tokens
top_k = 10
ndx_branch_root = masked_index
print(tokenizer.decode(torch.argsort(outputs[1][0,ndx_branch_root]).tolist()[-1:-top_k-1:-1]))
print('')

top_k = 20
for n,tok in enumerate(tokenized_text):
    print('[{}] '.format(tok),tokenizer.decode(torch.argsort(prediction_scores[:, n].squeeze(0)).tolist()[-1:-top_k - 1:-1]))
