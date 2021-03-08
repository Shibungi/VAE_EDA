from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class VAE():
    def __init__(self,input_dim,intermediate_dim,latent_dim,epochs):
        self.batch_size = 16
        self.epochs = epochs
        self.seed = 0
        torch.manual_seed(self.seed)
        self.device = torch.device("cpu")
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        self.network = VAE.network(input_dim,intermediate_dim,latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def train(self,train_data):
        self.network.train()
        tensor_data = torch.from_numpy(train_data).float()
        train_ = torch.utils.data.TensorDataset(tensor_data,tensor_data)
        train_loader = torch.utils.data.DataLoader(train_,batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.network(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            # print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    def sample(self,sample_num):
        with torch.no_grad():
            sample = torch.randn(sample_num, self.latent_dim).to(self.device)
            sample = self.network.decode(sample).cpu()
            return sample.to('cpu').detach().numpy().copy()

    class network(nn.Module):
        def __init__(self,input_dim,intermediate_dim,latent_dim):
            super(VAE.network, self).__init__()

            self.input_dim = input_dim
            self.intermediate_dim = intermediate_dim
            self.latent_dim = latent_dim

            # encoder
            self.fc1 = nn.Linear(input_dim, intermediate_dim)
            self.fc21 = nn.Linear(intermediate_dim, latent_dim)
            self.fc22 = nn.Linear(intermediate_dim, latent_dim)

            # decoder
            self.fc3 = nn.Linear(latent_dim, intermediate_dim)
            self.fc4 = nn.Linear(intermediate_dim, input_dim)

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std

        def decode(self, z):
            h3 = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))

        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, self.input_dim))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

#　層が多い
class VAE2():
    def __init__(self,input_dim,intermediate_dim,latent_dim,epochs):
        self.batch_size = 16
        self.epochs = epochs
        self.seed = 0
        torch.manual_seed(self.seed)
        self.device = torch.device("cpu")
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        self.network = VAE2.network(input_dim,intermediate_dim,latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def train(self,train_data):
        self.network.train()
        tensor_data = torch.from_numpy(train_data).float()
        train_ = torch.utils.data.TensorDataset(tensor_data,tensor_data)
        train_loader = torch.utils.data.DataLoader(train_,batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.network(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()

    def sample(self,sample_num):
        with torch.no_grad():
            sample = torch.randn(sample_num, self.latent_dim).to(self.device)
            sample = self.network.decode(sample).cpu()
            return sample.to('cpu').detach().numpy().copy()

    class network(nn.Module):
        def __init__(self,input_dim,intermediate_dim,latent_dim):
            super(VAE2.network, self).__init__()

            self.input_dim = input_dim
            self.intermediate_dim = intermediate_dim
            self.latent_dim = latent_dim

            # encoder
            self.fc1 = nn.Linear(input_dim, intermediate_dim)
            self.fc2 = nn.Linear(intermediate_dim, intermediate_dim)
            self.fc31 = nn.Linear(intermediate_dim, latent_dim)
            self.fc32 = nn.Linear(intermediate_dim, latent_dim)

            # decoder
            self.fc3 = nn.Linear(latent_dim, intermediate_dim)
            self.fc4 = nn.Linear(intermediate_dim, intermediate_dim)
            self.fc5 = nn.Linear(intermediate_dim, input_dim)

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))
            return self.fc31(h2), self.fc32(h2)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std

        def decode(self, z):
            h3 = F.relu(self.fc3(z))
            h4 = F.relu(self.fc4(h3))
            return torch.sigmoid(self.fc5(h4))

        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, self.input_dim))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

# 提案手法用VAE
class VAE3():
    def __init__(self,input_dim,intermediate_dim,latent_dim,epochs):
        self.batch_size = 16
        self.epochs = epochs
        self.seed = 0
        torch.manual_seed(self.seed)
        self.device = torch.device("cpu")
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        self.network = VAE.network(input_dim,intermediate_dim,latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def train(self,train_data):
        self.network.train()
        tensor_data = torch.from_numpy(train_data).float()
        train_ = torch.utils.data.TensorDataset(tensor_data,tensor_data)
        train_loader = torch.utils.data.DataLoader(train_,batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.network(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            # print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    def sample(self,sample_num,c):
        with torch.no_grad():
            sample = c * torch.randn(sample_num, self.latent_dim).to(self.device)
            sample = self.network.decode(sample).cpu()
            return sample.to('cpu').detach().numpy().copy()

    class network(nn.Module):
        def __init__(self,input_dim,intermediate_dim,latent_dim):
            super(VAE.network, self).__init__()

            self.input_dim = input_dim
            self.intermediate_dim = intermediate_dim
            self.latent_dim = latent_dim

            # encoder
            self.fc1 = nn.Linear(input_dim, intermediate_dim)
            self.fc21 = nn.Linear(intermediate_dim, latent_dim)
            self.fc22 = nn.Linear(intermediate_dim, latent_dim)

            # decoder
            self.fc3 = nn.Linear(latent_dim, intermediate_dim)
            self.fc4 = nn.Linear(intermediate_dim, input_dim)

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std

        def decode(self, z):
            h3 = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))

        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, self.input_dim))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar


## ---------------------------------------------------------------- 
class EVAE():
    def __init__(self,input_dim,intermediate_dim,latent_dim,epochs):
        self.batch_size = 16
        self.epochs = epochs
        self.seed = 0
        torch.manual_seed(self.seed)
        self.device = torch.device("cpu")
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        self.network = EVAE.network(input_dim,intermediate_dim,latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def loss_function(self,recon_x, x, recon_f, f, mu, logvar):
        # import pdb; pdb.set_trace()
        f = torch.sigmoid(f) # ここでfにsigmoidをかけている
        MSE = F.mse_loss(recon_f, f) # reduction='mean'
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD + MSE
    
    def train(self,train_data,train_score):
        self.network.train()
        tensor_data = torch.from_numpy(train_data).float()
        tensor_score = torch.from_numpy(train_score).float()
        train_ = torch.utils.data.TensorDataset(tensor_data,tensor_score)
        train_loader = torch.utils.data.DataLoader(train_,batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            train_loss = 0
            for batch_idx, (data, score) in enumerate(train_loader):
                data = data.to(self.device) 
                self.optimizer.zero_grad()
                ## ここまで
                recon_data_batch, recon_score_batch, mu, logvar = self.network(data)
                loss = self.loss_function(recon_data_batch, data, recon_score_batch, score, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            # print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
            

    def sample(self,sample_num):
        with torch.no_grad():
            sample = torch.randn(sample_num, self.latent_dim).to(self.device)
            sample = self.network.decode(sample).cpu()
            return sample.to('cpu').detach().numpy().copy()

    class network(nn.Module):
        def __init__(self,input_dim,intermediate_dim,latent_dim):
            super(EVAE.network, self).__init__()

            self.input_dim = input_dim
            self.intermediate_dim = intermediate_dim
            self.latent_dim = latent_dim

            # encoder
            self.fc1 = nn.Linear(input_dim, intermediate_dim)
            self.fc21 = nn.Linear(intermediate_dim, latent_dim)
            self.fc22 = nn.Linear(intermediate_dim, latent_dim)

            # decoder
            self.fc3 = nn.Linear(latent_dim, intermediate_dim)
            self.fc4 = nn.Linear(intermediate_dim, input_dim)

            # predictor
            self.fc5 = nn.Linear(latent_dim, intermediate_dim)
            self.fc6 = nn.Linear(intermediate_dim, 1)

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std

        def decode(self, z):
            h3 = F.relu(self.fc3(z))
            return torch.sigmoid(self.fc4(h3))

        def predict(self, z):
            h5 = F.relu(self.fc5(z))
            return torch.sigmoid(self.fc6(h5))

        def forward(self, x):
            mu, logvar = self.encode(x.view(-1, self.input_dim))
            z = self.reparameterize(mu, logvar)
            return self.decode(z), self.predict(z), mu, logvar



## ---------------------------------------------------------------- 
class CEVAE():
    def __init__(self,input_dim,intermediate_dim,latent_dim,epochs):
        self.batch_size = 16
        self.epochs = epochs
        self.seed = 0
        torch.manual_seed(self.seed)
        self.device = torch.device("cpu")
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        self.network = CEVAE.network(input_dim,intermediate_dim,latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def loss_function(self,recon_x, x, recon_f, f, mu, logvar):
        # import pdb; pdb.set_trace()
        f = torch.sigmoid(f) # ここでfにsigmoidをかけている
        MSE = F.mse_loss(recon_f, f) # reduction='mean'
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD + MSE
    
    def train(self,train_data,train_score):
        self.network.train()
        tensor_data = torch.from_numpy(train_data).float()
        tensor_score = torch.from_numpy(train_score).float()
        train_ = torch.utils.data.TensorDataset(tensor_data,tensor_score)
        train_loader = torch.utils.data.DataLoader(train_,batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            train_loss = 0
            for batch_idx, (data, score) in enumerate(train_loader):
                # import pdb; pdb.set_trace()
                data = data.to(self.device)
                score = score.to(self.device)
                self.optimizer.zero_grad()
                ## ここまで
                recon_data_batch, recon_score_batch, mu, logvar = self.network(data,score) #CVAE
                loss = self.loss_function(recon_data_batch, data, recon_score_batch, score, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            # print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    def sample(self,sample_num,aim_f): #CVAE
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            sample = torch.randn(sample_num, self.latent_dim).to(self.device)
            f_tensor = torch.tensor(aim_f).view(sample_num,1)
            sample = self.network.decode(sample,f_tensor).cpu()
            return sample.to('cpu').detach().numpy().copy()

    class network(nn.Module):
        def __init__(self,input_dim,intermediate_dim,latent_dim):
            super(CEVAE.network, self).__init__()

            self.input_dim = input_dim
            self.intermediate_dim = intermediate_dim
            self.latent_dim = latent_dim

            # encoder
            self.fc1 = nn.Linear(input_dim, intermediate_dim)
            self.fc21 = nn.Linear(intermediate_dim, latent_dim)
            self.fc22 = nn.Linear(intermediate_dim, latent_dim)

            # decoder #CVAE
            self.fc3 = nn.Linear(latent_dim + 1, intermediate_dim)
            self.fc4 = nn.Linear(intermediate_dim, input_dim)

            # predictor
            self.fc5 = nn.Linear(latent_dim, intermediate_dim)
            self.fc6 = nn.Linear(intermediate_dim, 1)

        def encode(self, x):
            h1 = F.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std

        def decode(self, z, f): #CVAE
            # import pdb; pdb.set_trace()
            in_ = torch.empty((z.shape[0], self.latent_dim + 1))
            in_[:,:self.latent_dim] = z
            in_[:,self.latent_dim:] = torch.sigmoid(f)
            h3 = F.relu(self.fc3(in_))
            return torch.sigmoid(self.fc4(h3))

        def predict(self, z):
            h5 = F.relu(self.fc5(z))
            return torch.sigmoid(self.fc6(h5))

        def forward(self, x, f): #CVAE
            mu, logvar = self.encode(x.view(-1, self.input_dim))
            z = self.reparameterize(mu, logvar)
            return self.decode(z,f), self.predict(z), mu, logvar
