import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        # the number of layers(Number of recurrent layers)
        # batch_first:If True, then the input and output tensors are provided as (batch, seq, feature) 
        # I also added a dropout of default 0.1 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        
        # the linear layer that maps the hidden state output dimension 
        # to the vocab size we want as output
        self.hidden2out = nn.Linear(hidden_size, vocab_size)
        
        # initialize the hidden state (see code below)
        self.hidden = (torch.zeros(1, 1, self.hidden_size),torch.zeros(1, 1, self.hidden_size))


    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence
        # first we need to concatenate the features along with the caption to act as LSTM inputs
        # Note1 that before concatenation I removed the last column in the captions array as it represents the <end> word that will not be           # fed to the LSTM (look at the network architecture illustrated in the 1_Preliminaries notebook)
        # Note2 that I had to perform embedding on the captions but not the feature vector as it is already embedded in the encoding step
        captions = captions[:, :-1]
        captions = self.word_embeddings(captions)
        features = features.unsqueeze(1)
        lstm_inputs = torch.cat((features, captions), 1)
        outputs, hidden = self.lstm(lstm_inputs)
        
        # Convert LSTM outputs to word predictions
        outputs = self.hidden2out(outputs)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        out = list()
        for i in range(max_len):
            outputs, states = self.lstm(inputs, states)
            
            #print('lstm output shape ', outputs.shape)
            #print('lstm output after reshape ', outputs.view(outputs.size(0), -1).shape)
            
            outputs = self.hidden2out(outputs.view(outputs.size(0), -1))#lstm output reshaping to prepare for the linear layer (1x640)
            
            #print('linear output shape ', outputs.shape)
            #print('linear output ', outputs)
            
            max_values,max_index = outputs.max(1) #getting the index of the highest probability in the vocabulary (vocab size is 9955), note the out of the linear layer is (1x9955), note:max(1) means the maximum in each row while max(0) means the max in each col
            
            #print('max_index  ', max_index)
            #print('max_index shape ', max_index.shape)
            out.append(max_index.item()) # appending the indecies to get the prediced sentence
            #print('out before embdedding ',out)
            #print('out embedding shape before unsqueezing ',(self.word_embeddings(max_index)).shape)
            
            inputs = self.word_embeddings(max_index).unsqueeze(1) # embedding then reshaping the previous step output to be the next step input (embedding output is 1x512 and unsqueezing out is 1x1x512)
            
            #print('new inputs shape ', inputs.shape, '\n')
        return out