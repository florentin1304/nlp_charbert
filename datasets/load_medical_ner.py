from datasets import load_dataset

dataset = load_dataset("jnlpba")

# divide DatasetDict into train, validation, and test
val_test = dataset['validation']
val = val_test.shard(2, 0)
test = val_test.shard(2, 1)

dataset['validation'] = val
dataset['test'] = test

dataset.save_to_disk('datasets/medical_domain/ner')