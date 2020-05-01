from torch import nn
import torchaudio
import torch
import tqdm
import torch.optim as optim



class ThreeSAModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        
        super(ThreeSAModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers = num_layers, \
                          batch_first = True, dropout = 0.5, bidirectional = True)
        
        self.fc1 = nn.Linear(2*hidden_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) 
        self.final = nn.Linear(hidden_dim, output_dim)
        
    
    def forward(self, inp, hidden = None):
        
        # Use last hidden layer as encoded represenatation of audio spectrogram
        _, hidden_states = self.gru(inp, hidden) # ignore output
        #final_state = torch.cat([hidden_states[-2:]]).sum(dim=0) # Either by summing
        final_state = hidden_states[-2:].view(1, 2*self.hidden_dim) # Or by concatenating
        
        x = self.fc1(final_state)
        x = self.fc2(x)        
        return self.final(x)
        
        
        
class Model_2:

    def __init__(self, sentence_embedder, audio_preprocessor, input_dim=161, output_dim = 300,
                 epochs = 10, hidden_dim = 512, num_layers = 2, load = False, load_path = None):

        self.sentence_embedder = sentence_embedder
        self.audio_preprocessor = audio_preprocessor

        self.model = ThreeSAModel(input_dim,hidden_dim,output_dim,num_layers)
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(),lr = 0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
          self.model=self.model.to(self.device)

        if load:
            state = torch.load(load_path)
            self.epochs = state['epoch']
            self.model.load_state_dict(state['model_state_dict'])
            self.optimizer.load_state_dict(state['optimizer_state_dict'])


    def train(self, dataset, save_path = '', epochs = 10, save_freq = 2):

        #loss function
        def loss_fn(x,y):
            return 1 - nn.functional.cosine_similarity(x,y)

        last_epoch = 0

        for epoch in range(epochs):
            for x in tqdm.tqdm(dataset):
                self.model.zero_grad()
                inp = self.audio_preprocessor(x[0]).permute(0,2,1).cuda()
                y = torch.tensor(self.sentence_embedder(x[2])).cuda()
                yhat = self.model(inp)
                

                loss = loss_fn(y,yhat)
                loss.backward()
                self.optimizer.step()

                if epoch - last_epoch == save_freq:
                    last_epoch = epoch
                    savefile = path + '/3SAModel_' + str(epoch) + '.tar'
                    torch.save({    'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, savefile)

        print('\n Training Finished')

    def predict(self,inp):

        inp = self.audio_preprocessor(inp).permute(0,2,1)
        return self.model(inp)

    def labelize(self, transcript):

        return self.sentence_embedder(transcript)