import argparse
import math
import time
from tqdm import tqdm
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchnet import meter
from torch.utils.tensorboard import SummaryWriter

from models.test import Transformer, nopeak_mask, create_masks
from data.dataset import dataset

os.makedirs("checkpoints", exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.set_defaults(train=True)
parser.add_argument('--test_epoch', type=int, default=2, help="the number of model to test")
# Train paramater
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=80, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=50, help="epoch from which to start lr decay")
parser.add_argument("--data_path", type=str, default="data/data/", help="path of the dataset")
parser.add_argument("--device", type=str, default="cuda:1", help="witch gpu to use")
# Model paramater
parser.add_argument("--enc_emb_dim", type=int, default=256, help="encoder embbeding dim")
parser.add_argument("--dec_emb_dim", type=int, default=256, help="decoder embbeding dim")
parser.add_argument("--enc_hid_dim", type=int, default=512, help="hidden dim for enceder")
parser.add_argument("--dec_hid_dim", type=int, default=512, help="hidden dim for decoder")
parser.add_argument("--attn_dim", type=int, default=32, help="attention dim")
parser.add_argument("--n_layers", type=int, default=2, help="number of hidden layers in encoder/decoder")
parser.add_argument("--max_len", type=int, default=50, help="max length for decoder output")
parser.add_argument("--enc_dropout", type=int, default=0.5, help="encoder dropout ratio")
parser.add_argument("--dec_dropout", type=int, default=0.5, help="decoder dropout ratio")
parser.add_argument("--input_dim", type=int, default=30000, help="maxsize of input vocabulary")
parser.add_argument("--output_dim", type=int, default=30000, help="maxsize of output vocabulary")
parser.add_argument("--clip", type=int, default=1, help="clipping gradient")
opt = parser.parse_args()

torch.cuda.set_device(1)
device = torch.device(opt.device) #if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count:,} trainable parameters')
    return

def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip, pad_id):
    model.train()
    loss_meter = meter.AverageValueMeter()
    for batch in tqdm(iterator):
        src = batch.src.t().contiguous()
        trg = batch.trg.t().contiguous()
        trg_input = trg[:, :-1]
        src_mask, trg_mask = create_masks(src, trg_input, pad_id, pad_id)
        optimizer.zero_grad()
        output = model(src, trg_input, src_mask, trg_mask)
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        loss_meter.add(loss.item())

    return loss_meter.value()[0]


@torch.no_grad()
def evaluate(model,iterator,criterion):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    loss_meter = meter.AverageValueMeter()

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        output = model(src, trg, 0) #turn off teacher forcing
        output = output.view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss_meter.add(loss.item())
    return loss_meter.value()[0]

@torch.no_grad()
def test(model,iterator,criterion,vocab):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    loss_meter = meter.AverageValueMeter()
    f = open('results/pred.txt', 'w+')
    ft = open('results/candidate.txt', 'w+')
    fr = open('results/reference.txt', 'w+')
    for i, batch in enumerate(iterator):
        src = batch.src.t().contiguous()
        trg = batch.trg.t().contiguous()

        src_mask = (src != vocab.stoi['<pad>']).unsqueeze(-2)
        e_outputs = model.encoder(src, src_mask)
        outputs = torch.zeros((src.shape[0], opt.max_len)).type_as(src.data)
        outputs[:, 0] = torch.LongTensor([vocab.stoi['<sos>']])

        for i in range(1, opt.max_len):
            trg_mask = np.triu(np.ones((1, i, i)),k=1).astype('uint8')
            trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()
            output = model.out(model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask))
            outputs[:,i] = output[:,-1,:].max(1)[1]

        output_text = outputs
        src_text = src
        trg_text = trg
        for (output_item, src_item, trg_item) in zip(output_text, src_text, trg_text):
            query = ''
            result = ''
            target = ''
            for (i, j, k) in zip(output_item[1:], src_item[1:], trg_item[1:]):
                result = result + vocab.itos[i] + ' '
                query = query + vocab.itos[j] + ' '
                target = target + vocab.itos[k] + ' '
            result = result.split('<eos>')[0]
            query = query.split('<eos>')[0]
            target = target.split('<eos>')[0]
            f.write('q:'+query+'\n')
            f.write('a:'+result+'\n')
            f.write('\n')
            ft.write(result+'\n')
            fr.write(target+'\n')
        
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)
        #loss = criterion(output, trg)
        #loss_meter.add(loss.item())

        
    return 1#loss_meter.value()[0]

if __name__ == '__main__':
    # Init dataset
    data = dataset(opt.data_path, opt.batch_size, device)
    train_iterator, test_iterator, valid_iterator = data.build()
    opt.input_dim = len(data.get_vocab())
    opt.output_dim = opt.input_dim
    # Init model self, src_vocab, trg_vocab, d_model, N_layers, heads, dropout):
    model = Transformer(opt.input_dim, opt.output_dim, 512, 6, 8, 0.1).to(device)
    if opt.train:
        #tensorboard
        writer = SummaryWriter(time.strftime('runs/'+"transformer"+'%m-%d-%H:%M'))
        #init wight
        def init_weights(m: nn.Module):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0, std=0.01)
                else:
                    nn.init.constant_(param.data, 0)
        #model.apply(init_weights)
        optimizer = optim.Adam(model.parameters(), lr = opt.lr, betas=(opt.b1, opt.b2))
        vocab = data.get_vocab()
        PAD_IDX = vocab.stoi['<pad>']
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        best_valid_loss = float('inf')
        for epoch in range(opt.n_epochs):
            print('Epoch:{}'.format(epoch))
            start_time = time.time()

            train_loss = train(model, train_iterator, optimizer, criterion, opt.clip, PAD_IDX)
            #valid_loss = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            #writer.add_scalars('loss',{'train': train_loss, 'val': valid_loss}, epoch)
            #writer.add_scalars('PPL',{'train': math.exp(train_loss), 'val': math.exp(valid_loss)}, epoch)
            writer.close()

            print(f'Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            torch.save(model.state_dict(), 'checkpoints/model{}.pkl'.format(epoch))
    else:
        model.load_state_dict(torch.load('checkpoints/model%d.pkl' % opt.test_epoch))
        vocab = data.get_vocab()
        PAD_IDX = vocab.stoi['<pad>']
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        print('loaded model')
        test_loss = test(model, valid_iterator, criterion, vocab)
        print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

