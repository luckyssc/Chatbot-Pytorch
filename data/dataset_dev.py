# coding:utf8
from torchtext.datasets import TranslationDataset
from torchtext.data import Field
from torchtext.data import BucketIterator
import torch
import spacy

class dataset():
    def __init__(self, path, batch_size, device):
        self.path = path
        self.batch_size = batch_size
        self.device = device

    def build(self):
        TEXT = Field(tokenize="spacy",
                    tokenizer_language="en",
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)
        datafields = [("src", TEXT), ("trg", TEXT)]
        train=TranslationDataset(path = self.path, exts = ("dev.enc","dev.dec"),
                                    fields = datafields)
        test=TranslationDataset(path = self.path, exts = ("dev.enc","dev.dec"),
                                    fields = datafields)
        val = TranslationDataset(path=self.path, exts=("dev.enc", "dev.dec"),
                                  fields=datafields)

        TEXT.build_vocab(train,test, min_freq = 2)
        self.field = TEXT
        self.vocab = TEXT.vocab
        self.padindex = TEXT.vocab.stoi['<pad>']
        #print(TEXT.vocab.stoi)
        train_iter, test_iter, val_iter = BucketIterator.splits((train, test, val),
                                                                batch_size=self.batch_size, device=self.device,
                                                                )
        return train_iter, test_iter, val_iter
    def get_padindex(self):
        return self.padindex

    def get_vocab(self):
        return self.vocab

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    data_set = dataset(path='data/', batch_size=128,
                       device=device)
    train_iterator, test_iterator, val_iterator = data_set.create()
    vocab = data_set.get_vocab()
    print(vocab.itos[4])

