# chatbot

[![requirments](https://img.shields.io/badge/pytorch-1.2.0-blue)](requirments)
[![requirments](https://img.shields.io/badge/torchtext-0.4.0-green)](requirments)

A base Seq2Seq+Attention chatbot model, using DD(daily dialogue) for dataset. 

## Content

- [Requirments](#Requirements)
- [Train](#Train)
- [Predict](#Predict)

## Requirements

- python 3.6+
- torch 1.2.0+
- torchtext 0.4.0+
- torchnet
- tensorboard

## Train

Train the model for default setting

```
python main.py --train
```

Or chage the default setting.

```
python main.py --help
usage: main.py [-h] [--train] [--test] [--epoch EPOCH] [--n_epochs N_EPOCHS]
               [--batch_size BATCH_SIZE] [--lr LR] [--b1 B1] [--b2 B2]
               [--decay_epoch DECAY_EPOCH] [--data_path DATA_PATH]
               [--device DEVICE] [--enc_emb_dim ENC_EMB_DIM]
               [--dec_emb_dim DEC_EMB_DIM] [--enc_hid_dim ENC_HID_DIM]
               [--dec_hid_dim DEC_HID_DIM] [--attn_dim ATTN_DIM]
               [--n_layers N_LAYERS] [--max_len MAX_LEN]
               [--enc_dropout ENC_DROPOUT] [--dec_dropout DEC_DROPOUT]
               [--input_dim INPUT_DIM] [--output_dim OUTPUT_DIM]
               [--clip CLIP]
```

The trained model will save in checkpoints/ folder named "model{n}.pkl" where the n is the number of epoch. And the tensorboard log file created in runs/ folder

## Predict

Predict the setences use trained model with test data.

```
python main --test
``` 

The results will be generate in results/ folder.

And the example of pred.txt is as follow

```
q:thank you very much . 
a:you 're welcome . 

q:ok . thank you . 
a:you are welcome . 

q:you 're looking great . 
a:yeah , i 'm not sure 

q:how 's the chicken ? 
a:i 'm not sure . 

q:one hundred dollars . 
a:here 's your money . i 
```
