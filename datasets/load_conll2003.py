import wget
import os

os.makedirs(os.path.dirname("./CoNLL2003/"), exist_ok=True)
train_file = wget.download("https://raw.githubusercontent.com/chnsh/BERT-NER-CoNLL/master/data/train.txt", "./CoNLL2003/train.txt")
val_file = wget.download("https://raw.githubusercontent.com/chnsh/BERT-NER-CoNLL/master/data/valid.txt", "./CoNLL2003/val.txt")
test_file = wget.download("https://raw.githubusercontent.com/chnsh/BERT-NER-CoNLL/master/data/test.txt", "./CoNLL2003/test.txt")
