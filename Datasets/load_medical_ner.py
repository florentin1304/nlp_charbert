from datasets import load_dataset
import os 

dataset = load_dataset("jnlpba")

# divide DatasetDict into train, validation, and test
val_test = dataset['validation']
val = val_test.shard(2, 0)
test = val_test.shard(2, 1)

dataset['validation'] = val
dataset['test'] = test

int_to_key = dataset['train'].features["ner_tags"].feature.int2str

path = 'datasets/medical_domain/ner/'

if not os.path.exists(path):
    os.makedirs(path)

with open('train.txt', 'w') as f:
    all_tokens = dataset['train']['tokens']
    all_tags = dataset['train']['ner_tags']
    for tokens, tags in zip(all_tokens, all_tags):
        for token, tag in zip(tokens, tags):
            f.write(f'{token} {int_to_key(tag)}\n')
        f.write('\n')

# create a txt file for validation set
with open('val.txt', 'w') as f:
    all_tokens = dataset['validation']['tokens']
    all_tags = dataset['validation']['ner_tags']
    for tokens, tags in zip(all_tokens, all_tags):
        for token, tag in zip(tokens, tags):
            f.write(f'{token} {int_to_key(tag)}\n')
        f.write('\n')

# create a txt file for test set
with open('test.txt', 'w') as f:
    all_tokens = dataset['test']['tokens']
    all_tags = dataset['test']['ner_tags']
    for tokens, tags in zip(all_tokens, all_tags):
        for token, tag in zip(tokens, tags):
            f.write(f'{token} {int_to_key(tag)}\n')
        f.write('\n')