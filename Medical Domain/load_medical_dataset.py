import datasets

dataset = datasets.load_dataset("argilla/medical-domain", "medical-domain")
dataset = dataset["train"].shuffle(seed = 42).select(range(len(dataset["train"]))) 
full = dataset["text"]

train_len = len(full)
val_len = test_len = int(train_len * 0.1)
train_len = train_len - val_len - test_len
train = full[:train_len]
val = full[train_len:train_len+val_len]
test = full[train_len+val_len:]

# save as txt file
with open('train_medical_domain_full.txt', 'w') as f:
    for text in train:
        f.write(text + '\n')
with open('val_medical_domain_full.txt', 'w') as f:
    for text in val:
        f.write(text + '\n')
with open('test_medical_domain_full.txt', 'w') as f:
    for text in test:
        f.write(text + '\n')