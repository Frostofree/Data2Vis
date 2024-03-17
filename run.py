# %%
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv 
import time
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %% [markdown]
# ## Preprocessing

# %%
### Create Vocabulary from ./Data/A3 files/train.targets train.sources dev.targets dev.sources test.targets test.sources

train_sources_text = open("./Data/A3 files/train.sources", "r")
train_targets_text = open("./Data/A3 files/train.targets", "r")
dev_sources_text = open("./Data/A3 files/dev.sources", "r")
dev_targets_text = open("./Data/A3 files/dev.targets", "r")
test_sources_text = open("./Data/A3 files/test.sources", "r")
test_targets_text = open("./Data/A3 files/test.targets", "r")

train_sources = train_sources_text.readlines()
train_targets = train_targets_text.readlines()
dev_sources = dev_sources_text.readlines()
dev_targets = dev_targets_text.readlines()
test_sources = test_sources_text.readlines()
test_targets = test_targets_text.readlines()

train_vocab = set()
test_vocab = set()


train_sources_text.close()
train_targets_text.close()
dev_sources_text.close()
dev_targets_text.close()
test_sources_text.close()
test_targets_text.close()

train_sources_list = []
train_targets_list = []
dev_sources_list = []
dev_targets_list = []
test_sources_list = []
test_targets_list = []

for line in train_sources:
    # Get individual charecters from line, add unique charecters to vocab
    train_vocab.update(set(line))
    # Add line to list after stripping \n
    train_sources_list.append(line.strip('\n'))

for line in train_targets:
    # Get individual charecters from line, add unique charecters to vocab
    test_vocab.update(set(line))
    # Add line to list after stripping \n
    train_targets_list.append(line.strip('\n'))

for line in dev_sources:
    # Get individual charecters from line, add unique charecters to vocab
    # vocab.update(set(line))
    # Add line to list after stripping \n
    dev_sources_list.append(line.strip('\n'))

for line in dev_targets:
    # Get individual charecters from line, add unique charecters to vocab
    # vocab.update(set(line))
    # Add line to list after stripping \n
    dev_targets_list.append(line.strip('\n'))

for line in test_sources:
    # Get individual charecters from line, add unique charecters to vocab
    # vocab.update(set(line))
    # Add line to list after stripping \n
    test_sources_list.append(line.strip('\n'))

for line in test_targets:
    # Get individual charecters from line, add unique charecters to vocab
    # vocab.update(set(line))
    # Add line to list after stripping \n
    test_targets_list.append(line.strip('\n'))


# Add <pad> and <SOS> , <EOS> to vocab
train_vocab.add('<pad>')
train_vocab.add('<SOS>')
train_vocab.add('<EOS>')

# Remove \n from vocab
train_vocab.remove('\n')

# Add <pad> and <SOS> , <EOS> to vocab
test_vocab.add('<pad>')
test_vocab.add('<SOS>')
test_vocab.add('<EOS>')

# Remove \n from vocab
test_vocab.remove('\n')


# %% [markdown]
# ### Creating Dictionary from Dataset

# %%
## create vocab to index and index to vocab dictionaries
source_char_to_int = {}
source_int_to_char = {}

for i, word in enumerate(train_vocab):
    source_char_to_int[word] = i
    source_int_to_char[i] = word

# print(source_char_to_int)

target_char_to_int = {}
target_int_to_char = {}

for i, word in enumerate(test_vocab):
    target_char_to_int[word] = i
    target_int_to_char[i] = word

print(target_char_to_int)

print(len(target_char_to_int))
print(len(source_char_to_int))

# %% [markdown]
# ### Split by Charecters

# %%
def encode_data_sources(data):
    encoded_data = []
    for i in range(len(data)):                  # appending 0 for <SOS> token 
        encoded_data.append([source_char_to_int[char] for char in data[i]])
    
    # encoded_data = [[[0]]] + encoded_data         # appending 0 for <SOS> token
    return encoded_data

def encode_data_targets(data):
    encoded_data = []
    for i in range(len(data)):                  # appending 0 for <SOS> token 
        encoded_data.append([target_char_to_int[char] for char in data[i]])
    
    # encoded_data = [[[0]]] + encoded_data         # appending 0 for <SOS> token
    return encoded_data

def decode_data_sources(data):
    decoded_data = []
    for i in range(len(data)+1):
        if i == 0:
            continue

        decoded_data.append([source_int_to_char[int] for int in data[i]])
    return decoded_data

def decode_data_targets(data):
    decoded_data = []
    for i in range(len(data)+1):
        if i == 0:
            continue

        decoded_data.append([target_int_to_char[int] for int in data[i]])
    return decoded_data


train_sources_encoded = encode_data_sources(train_sources_list)
train_targets_encoded = encode_data_targets(train_targets_list)
dev_sources_encoded = encode_data_sources(dev_sources_list)
dev_targets_encoded = encode_data_targets(dev_targets_list)
test_sources_encoded = encode_data_sources(test_sources_list)
test_targets_encoded = encode_data_targets(test_targets_list)

### For every sequence in train_sources_encoded, train_targets_encoded, dev_sources_encoded, dev_targets_encoded, 
# test_sources_encoded, test_targets_encoded
#  add <SOS> token at start of sequence 
# add <EOS> token at the end of the sequence and <pad> 
# tokens to make the sequence length equal to the maximum sequence length in the dataset which is 500

max_len = 500

for i in range(len(train_sources_encoded)):
    train_sources_encoded[i] = [source_char_to_int['<SOS>']] + train_sources_encoded[i] 
    train_targets_encoded[i].append(target_char_to_int['<EOS>'])

    if len(train_sources_encoded[i]) > max_len:
        train_sources_encoded[i] = train_sources_encoded[i][:max_len]       # Truncating the sequence to max_len
        train_targets_encoded[i] = train_targets_encoded[i][:max_len]       # Truncating the sequence to max_len
    else:
        train_sources_encoded[i] = train_sources_encoded[i] + [source_char_to_int['<pad>']] * (max_len - len(train_sources_encoded[i]))
        train_targets_encoded[i] = train_targets_encoded[i] + [target_char_to_int['<pad>']] * (max_len - len(train_targets_encoded[i]))

for i in range(len(dev_sources_encoded)):
    dev_sources_encoded[i] = [source_char_to_int['<SOS>']] + dev_sources_encoded[i] 
    dev_targets_encoded[i].append(target_char_to_int['<EOS>'])
    
    if len(dev_sources_encoded[i]) > max_len:
        dev_sources_encoded[i] = dev_sources_encoded[i][:max_len]       # Truncating the sequence to max_len
        dev_targets_encoded[i] = dev_targets_encoded[i][:max_len]       # Truncating the sequence to max_len
    else:
        dev_sources_encoded[i] = dev_sources_encoded[i] + [source_char_to_int['<pad>']] * (max_len - len(dev_sources_encoded[i]))
        dev_targets_encoded[i] = dev_targets_encoded[i] + [target_char_to_int['<pad>']] * (max_len - len(dev_targets_encoded[i]))

for i in range(len(test_sources_encoded)):
    test_sources_encoded[i] = [source_char_to_int['<SOS>']] + test_sources_encoded[i] 
    test_targets_encoded[i].append(target_char_to_int['<EOS>'])
    
    if len(test_sources_encoded[i]) > max_len:
        test_sources_encoded[i] = test_sources_encoded[i][:max_len]       # Truncating the sequence to max_len
        test_targets_encoded[i] = test_targets_encoded[i][:max_len]       # Truncating the sequence to max_len
    else:
        test_sources_encoded[i] = test_sources_encoded[i] + [source_char_to_int['<pad>']] * (max_len - len(test_sources_encoded[i]))
        test_targets_encoded[i] = test_targets_encoded[i] + [target_char_to_int['<pad>']] * (max_len - len(test_targets_encoded[i]))

print(len(train_sources_encoded[0]))


# %% [markdown]
# ### Convert to Tensors

# %%
train_sources_tensor = torch.Tensor(train_sources_encoded)
train_targets_tensor = torch.Tensor(train_targets_encoded)
dev_sources_tensor = torch.Tensor(dev_sources_encoded)
dev_targets_tensor = torch.Tensor(dev_targets_encoded)
test_sources_tensor = torch.Tensor(test_sources_encoded)
test_targets_tensor = torch.Tensor(test_targets_encoded)


print(train_sources_tensor.shape)
print(train_targets_tensor.shape)
print(dev_sources_tensor.shape)
print(dev_targets_tensor.shape)
print(test_sources_tensor.shape)
print(test_targets_tensor.shape)

# %% [markdown]
# ### Creating Dataset and Dataloaders

# %%

class MyDataset(Dataset):
    def __init__(self, sources, targets):
        self.sources = sources
        self.targets = targets
    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = self.sources[idx]
        target = self.targets[idx]
        return source, target
    
train_dataset = MyDataset(train_sources_tensor, train_targets_tensor)
dev_dataset = MyDataset(dev_sources_tensor, dev_targets_tensor)
test_dataset = MyDataset(test_sources_tensor, test_targets_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# %% [markdown]
# ### Creating Seq2Seq Model with Attention

# %%
### Seq2Seq with Attention

## Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size=512, hidden_size=512, num_layers=2, dropout=0.5,bidirectional=True,batch_first=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=batch_first)

    def encode_input(self, x):
        encoded_x = torch.zeros(len(x),500,dtype= int)
        for i in range(len(x)):
            for j in range(len(x[i])):
                print(x[i][j].cpu().numpy())
                encoded_x[i][j] = source_char_to_int[int(x[i][j].cpu().numpy())]
            encoded_x[i][len(x[i])] = source_char_to_int['<EOS>']
        
        return encoded_x.to(device)

    def forward(self, x):
        # x shape: (batch_size, seq_length)

        # encoded_x = self.encode_input(x)

        embedding = self.dropout(self.embedding(x))                     # As per assignment, dropout is applied to embedding and not to inputs of hidden layer

        # embedding shape: (batch_size, seq_length, embedding_size)

        outputs, (hidden, cell) = self.lstm(embedding)

        # print("Outputs (Enc)",outputs.shape)

        # outputs shape: (N, 500, 1024)
        # hidden shape: (num_layers*num_directions, batch_size, hidden_size)
        # cell shape: (num_layers*num_directions, batch_size, hidden_size)

        # concatenate hiden states of last layer of bidrectional LSTM
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        # hidden shape: (2,N,1024)

        return outputs, hidden, cell
    

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        # self.linear = nn.Linear()

    
    def forward(self,decoder_hidden, encoder_outputs):
        # decoder_hidden shape: (2, N, 1024)
        # encoder_outputs shape: (N, 500, 1024)

        # attention shape: (N,500)
        hidden_last = decoder_hidden[-1,:,:].unsqueeze(0)
        # print("Hidden last", hidden_last.shape)
                

        # print(encoder_outputs.shape)
        attention = torch.matmul(hidden_last.permute(1,0,2),encoder_outputs.permute(0,2,1))

        attention = self.softmax(attention)

        context = torch.matmul(attention,encoder_outputs)

        # context shape: (N,1,1024)

        return context, attention
    
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size=512, hidden_size=512, num_layers=2, dropout=0.5, batch_first=True):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size+hidden_size*2, hidden_size*2, num_layers, dropout=dropout, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.attention = Attention()
    
    def forward(self, x, encoder_hidden, encoder_outputs,teacher_forcing_ratio=1.0):
        ### x : (N,500)
        ### hidden : (2,N,1024)
        ### encoder_outputs : (N,500,1024)
        ### teacher_forcing_ratio : float

        batch_size = x.shape[0]
        max_len = x.shape[1]
        # print("Encoder outputs (Dec)",encoder_outputs.shape)
        vocab_size = self.output_size

        target_embedding = self.dropout(self.embedding(x))
        # target_embedding shape: (N,500,512)

        initial_hidden = torch.randn(self.num_layers, batch_size, self.hidden_size*2).to(device)    # (2,N,1024)
        initial_cell = torch.randn(self.num_layers, batch_size, self.hidden_size*2).to(device)      # (2,N,1024)

        outputs = []
        query = []
        cell_states = []
        hidden_states = [initial_hidden]            # (2,N,1024)

        for timestep in range(max_len):

            if (timestep == 0):
                context , attention = self.attention(hidden_states[-1],encoder_outputs)

                if np.random.random() < teacher_forcing_ratio:
                    input = target_embedding[:,timestep,:]
                    input = input.unsqueeze(1)
                else:
                    input = torch.tensor([target_char_to_int['SOS']]*batch_size).to(device) # (N)
                    input  = input.unsqueeze(1)         # (N,1)
                    input  = self.embedding(input)      # (N,1,512)
                
                # print("Input shape",input.shape)
                # print("Context shape",context.shape)
                
                input = torch.cat((input,context),dim=2)
                # input shape: (N,1,1536)
                # print("Input shape",input.shape)

                output, (dec_hidden, cell) = self.lstm(input, (initial_hidden, initial_cell))    # (N,1,1024)
                # print("Output shape",output.shape)
                # print("Hidden shape",hidden.shape)
                # print("Cell shape",cell.shape)
                # output shape: (N,1,1024)
                # hidden shape: (2,N,1024)
                # cell shape: (2,N,1024)
            else:

                context , attention = self.attention(hidden_states[-1],encoder_outputs)

                if np.random.random() < teacher_forcing_ratio:
                    input = target_embedding[:,timestep,:]
                    input = input.unsqueeze(1)

                else:
                    input = output[-1]
                    input = self.embedding(input)
                    # input shape: (N,1,512)
                
                input = torch.cat((input,context),dim=2)
                # input shape: (N,1,1536)

                # print("Input shape",input.shape)

                # hidden here is from the last time step of the last layer, not last layer itself

                output, (hidden, cell) = self.lstm(input, (hidden_states[-1], cell))    # (N,1,1024)
                # output shape: (N,1,1024)
                # hidden shape: (2,N,1024)
                # cell shape: (2,N,1024)
            
            # print("Output shape",output.shape)
            output = self.fc(output.squeeze(1))

            # output = F.softmax(output, dim=1)

            # output shape: (N, 500)
            outputs.append(output)
            hidden_states.append(dec_hidden)
            cell_states.append(cell)
            query.append(dec_hidden)        # previous dec_hidden

        
        outputs = torch.stack(outputs, dim=1)
        hidden_states = torch.stack(hidden_states, dim=1)
        cell_states = torch.stack(cell_states, dim=1)

        outputs = nn.LogSoftmax(dim=2)(outputs)
        
        return outputs, hidden_states, cell_states
    
    def predict(self,hidden,encoder_outputs):

        batch_size = encoder_outputs.shape[0]
        
        outputs = []
        hidden_states = []
        cell_states = []
        query = [hidden]            # (2,N,1024)

        input = torch.tensor([target_char_to_int['<SOS>']]*batch_size).unsqueeze(1).to(device)    # (N,1)

        for timestep in range(500):
            
            output, hidden , cell = self.forward(input,hidden,encoder_outputs,teacher_forcing_ratio=1.0)            # (N,2)

            output = F.softmax(output, dim=-1)

            outputs.append(output)
            
            output = torch.argmax(output,dim=-1)

            # Concatenate x with output
            # print("Output shape",output.shape)
            input = output
            # print("Input shape",input.shape)

            # output = output.detach()
            hidden_states.append(hidden)
            cell_states.append(cell)
        
        


        outputs = torch.stack(outputs, dim=1)

        outputs = torch.argmax(nn.Softmax(dim=2)(outputs),dim=2)
        # print("Final outputs",outputs.shape)
        hidden_states = torch.stack(hidden_states, dim=1)
        cell_states = torch.stack(cell_states, dim=1)

        return outputs, hidden_states, cell_states


# %%
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing=1.0):
        # source shape: (batch_size, seq_length)
        # target shape: (batch_size, seq_length)

        batch_size = source.shape[0]
        seq_length = source.shape[1]

        encoder_outputs, encoder_hidden, encoder_cell = self.encoder(source)
        
        # encoder_outputs shape: (batch_size, seq_length, hidden_size*num_directions)
        # encoder_hidden shape: (num_layers*num_directions, batch_size, hidden_size)
        # encoder_cell shape: (num_layers*num_directions, batch_size, hidden_size)

        decoder_outputs, decoder_hidden, attentions = self.decoder(target,encoder_hidden,encoder_outputs,teacher_forcing)
        # decoder_outputs shape: (batch_size, seq_length, hidden_size*num_directions*2)
        # decoder_hidden shape: (num_layers*num_directions, batch_size, hidden_size)
        # attentions shape: (batch_size, seq_length)

        return decoder_outputs, decoder_hidden, attentions

# %%
### Train function
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - elapsed_mins*60)
    return elapsed_mins, elapsed_secs


def train(model, criterion, optimizer, train_loader, dev_loader, num_epochs):
    train_losses = []
    dev_losses = []

    steps = 0
    eval_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        for i, (source, target) in enumerate(train_loader):
            source = source.long().to(device)
            target = target.long().to(device)

            optimizer.zero_grad()

            encoder_outputs, encoder_hidden, encoder_cell = model.encoder(source)
            # print("Encoder Outputs:",encoder_outputs.shape)
            # print("Encoder Hidden:",encoder_hidden.shape)
            # print("Encoder Cell:",encoder_cell.shape)
            
            decoder_outputs, decoder_hidden, attention_weights = model.decoder(target,encoder_hidden, encoder_outputs)

            # decoder_outputs shape: (batch_size, seq_length, hidden_size*num_directions*2)
            # target shape: (batch_size, seq_length)
            # decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[-1])
            # target = target.view(-1)
            # print("Decoder Outputs:",decoder_outputs.shape)
            # print("Target:",target.shape)

            loss = criterion(decoder_outputs.view(-1,46), target.view(-1))
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            perplexity = np.exp(loss.item())

            steps += 1
            print("Batch:",steps+1,"Loss:",loss.item(),"Perplexity:",perplexity)

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        dev_loss = 0
        with torch.no_grad():
            for i, (source, target) in enumerate(dev_loader):
                source = source.long().to(device)
                target = target.long().to(device)

                encoder_outputs, encoder_hidden, encoder_cell = model.encoder(source)
                decoder_outputs, decoder_hidden, decoder_cell = model.decoder(target,encoder_hidden, encoder_outputs,1.0)

                # decoder_outputs shape: (batch_size, seq_length, hidden_size*num_directions)
                # target shape: (batch_size, seq_length)
                # decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[-1])
                # target = target.view(-1)

                decoder_outputs = decoder_outputs.view(-1,46)
                target = target.view(-1)
                
                loss = criterion(decoder_outputs, target)
                dev_loss += loss.item()

        dev_loss /= len(dev_loader)
        dev_losses.append(dev_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s',f'\tTrain Loss: {train_loss:.3f}',f'\t Val. Loss: {dev_loss:.3f}' , f'\t Val. PPL: {np.exp(dev_loss):7.3f}')

        torch.save(model.state_dict(),"model_{epoch}.pt")
    

    return train_losses, dev_losses


### Test function
# def test(model, criterion, test_loader):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (source, target) in enumerate(test_loader):
#             source = source.long().to(device)
#             target = target.long().to(device)

#             encoder_outputs, encoder_hidden, encoder_cell = model.encoder(source)
#             decoder_outputs, decoder_hidden, decoder_cell = model.decoder.predict(encoder_hidden, encoder_outputs)

#             # print("decoder outputs shape",decoder_outputs.squeeze(2).shape)
#             # print("target shape",target.shape)

#             # decoder_outputs shape: (batch_size, seq_length, hidden_size*num_directions)
#             # target shape: (batch_size, seq_length)
#             # decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[-1])
#             # target = target.view(-1)

#             # loss = criterion(decoder_outputs.view(-1,46), target.view(-1))
#             # test_loss += loss.item()

#     # test_loss /= len(test_loader)
#     # print(f'Test Loss: {test_loss:.3f}')

#     return  , decoder_outputs

# %%
#### Training and Testing

## Hyperparameters
input_size = len(train_vocab)
output_size = len(test_vocab)
embedding_size = 512
hidden_size = 512
num_layers = 2
dropout = 0.5
bidirectional = True
batch_first = True
teacher_forcing_ratio = 1.0
num_epochs = 10

encoder = Encoder(input_size, embedding_size, hidden_size, num_layers, dropout, bidirectional, batch_first).to(device)
decoder = Decoder(output_size, embedding_size, hidden_size, num_layers, dropout, batch_first).to(device)

model = Seq2Seq(encoder, decoder,device)

criterion = nn.CrossEntropyLoss(ignore_index=target_char_to_int['<pad>'])
optimizer = optim.Adam(model.parameters(),lr=1e-4)

train_losses, dev_losses = train(model, criterion, optimizer, train_loader, dev_loader, num_epochs)


# %%
### Save Model
torch.save(model.state_dict(), './model_e0b1600.pt')

# %%
# Load Keys from Model
encoder = Encoder(input_size, embedding_size, hidden_size, num_layers, dropout, bidirectional, batch_first).to(device)
decoder = Decoder(output_size, embedding_size, hidden_size, num_layers, dropout, batch_first).to(device)
model = Seq2Seq(encoder, decoder,device)

model.load_state_dict(torch.load('./model.pt'))


# %% [markdown]
# ### Testing

# %%
def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (source, target) in enumerate(test_loader):
            source = source.long().to(device)
            target = target.long().to(device)

            encoder_outputs, encoder_hidden, encoder_cell = model.encoder(source)
            decoder_outputs, decoder_hidden, decoder_cell = model.decoder.predict(encoder_hidden, encoder_outputs)

            # print("decoder outputs shape",decoder_outputs.squeeze(2).shape)
            
            # print("target shape",target.shape)

            # decoder_outputs shape: (batch_size, seq_length, hidden_size*num_directions)
            # target shape: (batch_size, seq_length)
            # decoder_outputs = decoder_outputs.view(-1, decoder_outputs.shape[-1])
            # target = target.view(-1)

            # loss = criterion(decoder_outputs.squeeze(2).view(-1,46), target.view(-1))   # for test dataset
            # test_loss += loss.item()

    # test_loss /= len(test_loader)
    # print(f'Test Loss: {test_loss:.3f}')

    return test_loss , decoder_outputs

test_loss , _ = test(model, criterion, test_loader)

# %%
print(test_loss)

# %% [markdown]
# ### Local Beam Search

# %%
def calculate_diversity_penalty(new_sequence, existing_sequences):
    """
    Calculate a diversity penalty based on the new sequence and existing sequences.
    This is a simple example of diversity penalty calculation and can be customized.

    Args:
    - new_sequence (torch.Tensor): The new sequence to be penalized.
    - existing_sequences (list of torch.Tensor): A list of existing sequences.

    Returns:
    - float: The diversity penalty score.
    """
    penalty = 0.0
    for seq in existing_sequences:
        similarity = torch.sum(torch.eq(new_sequence, seq[0]).float()) / len(new_sequence)
        penalty += similarity
    return penalty

def beam_search_decoder(probabilities, beam_width, max_length, diversity_penalty_weight=0.7):
    """
    Beam search decoder for sequence generation.

    Args:
    - probabilities (torch.Tensor): A 2D tensor of shape (sequence_length, vocab_size)
      containing the predicted probabilities for each token at each time step.
    - beam_width (int): The number of sequences to consider at each decoding step.
    - max_length (int): The maximum length of the generated sequence.

    Returns:
    - List of tuples, each containing (sequence, score), where:
      - sequence (list): A list of token IDs representing the generated sequence.
      - score (float): The log-likelihood score of the sequence.
    """
    out  = torch.argmax(nn.Softmax(dim = 1)(probabilities), dim = 1)

    # out = out.squeeze(0)
    # out = out.squeeze(0)
    # print("Output shape",out.shape)
    # print(out)
    seq_len = 0
    for char in out:
        if(char == target_char_to_int["<EOS>"]):
            break
        else:
            seq_len += 1

    # Get the sequence length and vocabulary size
    sequence_length, vocab_size = probabilities.shape
    sequence_length = seq_len
    max_length = seq_len
    print(seq_len)

    # Initialize the beam with the empty sequence
    beam = [(torch.tensor([], dtype=torch.long).to(device), 0.0)]

    # Iterate through each time step
    for t in range(max_length):
        new_beam = []

        # Expand the beam by considering the top 'beam_width' candidates at each step
        for sequence, score in beam:
            # If the sequence is already at the maximum length, keep it as is
            if len(sequence) == max_length:
                new_beam.append((sequence, score))
                continue

            # Get the probabilities for the next token
            t_probs = probabilities[t]

            # Get the top 'beam_width' token IDs and their corresponding log-likelihood scores
            top_scores, top_tokens = torch.topk(t_probs, beam_width)

            # Expand the current sequence with each of the top tokens
            for token, token_score in zip(top_tokens, top_scores):
                new_sequence = torch.cat([sequence, token.unsqueeze(0)], dim=0)
                new_score = score + token_score.item()
    
                # Apply the diversity penalty
                if len(new_sequence) > 1:
                    # Calculate a penalty based on sequence diversity
                    diversity_penalty = diversity_penalty_weight * calculate_diversity_penalty(new_sequence, new_beam)
                    new_score -= diversity_penalty
                    
                new_beam.append((new_sequence, new_score))
        print(t)

        # Keep the top 'beam_width' candidates
        new_beam.sort(key=lambda x: -x[1])
        beam = new_beam[:beam_width]

    # Return the top sequence and its score
    return [(sequence.tolist(), score) for sequence, score in beam]
    


# %%
def convert_to_char(seq):
        vis = ""
        for char in seq:
            char = char
            if(char == '<EOS>'):
                return vis
            vis += target_int_to_char[char]
        
        return vis

# %% [markdown]
# ### Get Progression data

# %%
# ## Transform Progression data

with open("./Data/A3 files/progression.txt", "r") as f:
    progression_dev = f.readlines()

f.close()

transform = {}
transform["year"] = "num0"
transform["race"] = "str0"
transform["Time"] = "num1"
transform["Distance"] = "num2"

reverse_transform = {}
reverse_transform["num0"] = "year"
reverse_transform["str0"] = "race"
reverse_transform["num1"] = "Time"
reverse_transform["num2"] = "Distance"

progression_dev = [line.strip('\n') for line in progression_dev]

data = "[" + progression_dev[1] + "]"


# Create dataset using data

data = encode_data_sources(data)

data = [[source_char_to_int["<SOS>"]]] + data
data.append([source_char_to_int["<EOS>"]])

data = data + [[source_char_to_int['<pad>']]] * (500 - len(data))

data = torch.Tensor(data).long().reshape(1,-1)

progression_dataset = MyDataset(data,torch.Tensor([[0]]))

progression_loader = DataLoader(progression_dataset, batch_size=1, shuffle=False)

# %%
print(progression_loader.dataset.sources.shape)

# %%
criterion = nn.CrossEntropyLoss()
predictions , probs = test(model, criterion, progression_loader)


# print("1",probs.shape)
probs = probs.reshape(500,46)
# print("2",probs.shape)
# probabilities = probs.squeeze(0)
# print(probs.shape)

seq_and_score = beam_search_decoder(probs, 15, 500, diversity_penalty_weight=0.7)

# %%
print(seq_and_score[0][0])

print(target_char_to_int["<pad>"])

print(convert_to_char(seq_and_score[0][0]))


