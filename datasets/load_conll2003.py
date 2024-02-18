import wget
import os

os.makedirs(os.path.dirname("./CoNLL2003/"), exist_ok=True)
train_file = wget.download("https://github.com/pranabsarkar/Conll_task/blob/master/conll-2003/eng.train", "./CoNLL2003/train.txt")
val_file = wget.download("https://github.com/pranabsarkar/Conll_task/blob/master/conll-2003/eng.testa", "./CoNLL2003/val.txt")
test_file = wget.download("https://github.com/pranabsarkar/Conll_task/blob/master/conll-2003/eng.testb", "./CoNLL2003/test.txt")
