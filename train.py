import torch
import json
import csv
from tqdm import tqdm
from pprint import pprint
import numpy as np
from torch import nn, optim
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

filename = "train.en"
file1 = open(filename, 'r')
Lines = file1.readlines()
import regex as re
def preprocess(text):
    # remove all the punctuation marks
    text = text.lower()
    # remove all non alpha numeric characters
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    # remove all the extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data_list = []
for item in Lines:
    #print(item[:-1])
    string = preprocess(item[:-1])
    data_list.append(string)

filename = "train.fr"
file1 = open(filename, 'r')
Lines = file1.readlines()
data_list_fr = []
for item in Lines:
    #print(item[:-1])
    string = preprocess(item[:-1])
    data_list_fr.append(string)

# replace 10 % of the data with <unk>, english and french
for i in range(len(data_list)):
    if np.random.rand() < 0.1:
        data_list[i] = "<unk>"
for i in range(len(data_list_fr)):
    if np.random.rand() < 0.1:
        data_list_fr[i] = "<unk>"
        

# for each sent in eng, add sos and eos
# for each sent in fr, add sos and eos
for i in range(len(data_list)):
    data_list[i] = "<sos> " + data_list[i] + " <eos>"
    data_list_fr[i] = "<sos> " + data_list_fr[i] + " <eos>"


eng_len_list = []
fr_len_list = []
# pad all the sentences to the max length using <pad>
max_eng_len = 0
max_fr_len = 0
eng_final_list = []
fr_final_list = []
for item in data_list:
    loc = item.split()
    length = len(loc)
    eng_len_list.append(length)
    if length > max_eng_len:
        max_eng_len = length
    eng_final_list.append(loc)
for item in data_list_fr:
    loc = item.split()
    length = len(loc)
    fr_len_list.append(length)
    if length > max_fr_len:
        max_fr_len = length
    fr_final_list.append(loc)

# pad all the sentences to the max length using <pad>
for i in range(len(eng_final_list)):
    while len(eng_final_list[i]) < max_eng_len:
        eng_final_list[i].append("<pad>")
for i in range(len(fr_final_list)):
    while len(fr_final_list[i]) < max_fr_len:
        fr_final_list[i].append("<pad>")

eng_wrd2idx = {}
eng_idx2wrd = {}
fr_wrd2idx = {}
fr_idx2wrd = {}
for item in eng_final_list:
    for wrd in item:
        if wrd not in eng_wrd2idx:
            eng_wrd2idx[wrd] = len(eng_wrd2idx)
            eng_idx2wrd[len(eng_idx2wrd)] = wrd
for item in fr_final_list:
    for wrd in item:
        if wrd not in fr_wrd2idx:
            fr_wrd2idx[wrd] = len(fr_wrd2idx)
            fr_idx2wrd[len(fr_idx2wrd)] = wrd
eng_list = []
fr_list = []
# make the list of indices
for item in eng_final_list:
    loc = []
    for wrd in item:
        loc.append(eng_wrd2idx[wrd])
    eng_list.append(loc)
for item in fr_final_list:
    loc = []
    for wrd in item:
        loc.append(fr_wrd2idx[wrd])
    fr_list.append(loc)
    
class Eng2Fr_Dataset(torch.utils.data.Dataset):
    def __init__(self, eng_list, fr_list, eng_lens, fr_lens):
        self.eng_list = eng_list
        self.fr_list = fr_list
        self.eng_lens = eng_lens
        self.fr_lens = fr_lens
    def __len__(self):
        return len(self.eng_list)
    def __getitem__(self, idx):
        return torch.tensor(self.eng_list[idx]), torch.tensor(self.fr_list[idx]), self.eng_lens[idx], self.fr_lens[idx]

dataset = Eng2Fr_Dataset(eng_list, fr_list, eng_len_list, fr_len_list)   

dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
# make 10 % of the data as validation data
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=True)

from encoder.modules import *
from decoder.modules import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, vocab_size):
        super(Encoder, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_encoding = PosEncoding()
        self.encoding_layer1 = EncodingLayer(model_dim, num_heads, dropout)
        self.encoding_layer2 = EncodingLayer(model_dim, num_heads, dropout)
        self.linear_layer = nn.Linear(model_dim, model_dim)
    def forward(self, x, len_list):
        embedded = self.embedding(x)
        pos_encoded = self.pos_encoding(embedded).to(device)
        pos_encoded = apply_mask(pos_encoded, len_list).to(device)
        out1 = self.encoding_layer1(pos_encoded)
        out1 = apply_mask(out1, len_list).to(device)
        out2 = self.encoding_layer2(out1)
        out2 = apply_mask(out2, len_list).to(device)
        out3 = self.linear_layer(out2)
        out3 = apply_mask(out3, len_list).to(device)
        return out3
class Decoder(nn.Module):
    def __init__(self, model_dim, num_heads, dropout, vocab_size):
        super(Decoder, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.decoding_layer1 = DecodingLayer(model_dim, num_heads, dropout)
        self.decoding_layer2 = DecodingLayer(model_dim, num_heads, dropout)
        self.linear_layer = nn.Linear(model_dim, vocab_size)
    def forward(self, encoded, len_list, decoded):
        #print("decoded shape", decoded.shape)
        decoded = self.embedding(decoded)
        #print("after embeddings: ", decoded.shape)
        encoded = apply_mask(encoded, len_list)
        #print(decoded.shape, ">>>>>>>><<<<<<<<<<<<<<<<")
        decoded = self.decoding_layer1(encoded, decoded)
        encoded = apply_mask(encoded, len_list)
        #print(decoded.shape, ">>>>>>>><<<<<<<<<<<<<<<<")
        decoded = self.decoding_layer2(encoded, decoded)
        #print(decoded.shape, ">>>>>>>>")
        decoded = self.linear_layer(decoded)
        #print(decoded.shape, "<<<<<<<<<<<<<<<")
        return decoded

torch.autograd.set_detect_anomaly(True)
Encoder_model = Encoder(300, 3, 0.1, len(eng_wrd2idx))
Decoder_model = Decoder(300, 3, 0.1, len(fr_wrd2idx))
Encoder_model = torch.load("encoder_model.pt")
Decoder_model = torch.load("decoder_model.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Encoder_model.to(device)
Decoder_model.to(device)
# make the loss function ignore sos, eos, pad in english and french write criterion accordingly
criterion = nn.CrossEntropyLoss(ignore_index=eng_wrd2idx["<pad>"])
encoder_optimizer = optim.Adam(Encoder_model.parameters(), lr=0.00001)
decoder_optimizer = optim.Adam(Decoder_model.parameters(), lr=0.00001)
# enumerate through batches
def calculate_bleu(pred, truth):
    #print(pred.shape, truth.shape)
    #print(pred, truth)
    pred = pred.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()
    # convert to 1d
    pred = pred.flatten()
    truth = truth.flatten()
    # convert to int
    preds = []
    truths = []
    for item in pred:
        #print(item)
        preds.append(int(item))
    for item in truth:
        truths.append(int(item))
    #print(preds, truths)
    chencherry = SmoothingFunction()
    bleu_score = sentence_bleu([truths], preds, smoothing_function=chencherry.method1)
    return bleu_score
n_epochs = 10
loss_list = []
for epoch in range(n_epochs):
    for batch , (x, y, x_len, y_len) in enumerate(train_dataloader):
        #print(x.shape, y.shape, x_len.shape, y_len.shape)
        x = x.to(device)
        y = y.to(device)
        x_len = x_len.to(device)
        y_len = y_len.to(device)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoded = Encoder_model(x, x_len)
        total_loss = 0
        # tokens = max of y_len
        decoded  = Decoder_model(encoded, x_len, y)
        prediction = decoded.view(-1, decoded.size(-1))
        label = y.view(-1)
        #print(prediction.shape, label.shape)
        loss = criterion(prediction, label)
        loss.backward(retain_graph=True)
        encoder_optimizer.step()
        decoder_optimizer.step()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss_list.append(loss.item())
        print("Epoch: ", epoch, "Batch: ", batch, "Loss: ", loss.item())
    print("Entering Val Loop: ")
    # write val loop
    for batch, (x, y, x_len, y_len) in enumerate(val_dataloader):
        x = x.to(device)
        y = y.to(device)
        x_len = x_len.to(device)
        y_len = y_len.to(device)
        encoded = Encoder_model(x, x_len)
        decoded = Decoder_model(encoded, x_len, y)
        prediction = decoded.view(-1, decoded.size(-1))
        label = y.view(-1)
        loss = criterion(prediction, label)
        print("Epoch: ", epoch, "Val Batch: ", batch, "Loss: ", loss.item())
    #     print("Epoch: ", epoch, "Batch: ", batch, "Loss: ", total_loss)
    torch.save(Encoder_model, "encoder_model.pt")
    torch.save(Decoder_model, "decoder_model.pt")
    # savel en_wrd2idx, eng_idx2wrd, fr_wrd2idx, fr_idx2wrd
    torch.save(eng_wrd2idx, "eng_wrd2idx.pt")
    torch.save(eng_idx2wrd, "eng_idx2wrd.pt")
    torch.save(fr_wrd2idx, "fr_wrd2idx.pt")
    torch.save(fr_idx2wrd, "fr_idx2wrd.pt")
    torch.save(loss_list, "loss_list2.pt")

            
