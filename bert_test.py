#%%
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

tokens = tokenizer(["My name is Ted and I am a squid.", "I am seriously a squid!"],
                   padding='max_length', max_length = 50, 
                   truncation=True, return_tensors="pt")

decoded = tokenizer.decode(tokens.input_ids[0])

print(tokens['input_ids'])
print(decoded)
# %%
from torchinfo import summary as torch_summary
from transformers import BertModel

bert = BertModel.from_pretrained('bert-base-cased')
_, pooled_output = bert(input_ids= tokens['input_ids'], attention_mask=tokens['attention_mask'],return_dict=False)
print(pooled_output.shape)
print(pooled_output)
# %%
