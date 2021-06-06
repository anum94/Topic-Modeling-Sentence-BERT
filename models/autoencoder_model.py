import torch.nn.functional as F
import torch
from torch import nn


class model(nn.Module):
    """
    Autoencoder for learning latent space representation
    architecture simplified for only one hidden layer
    """

    def __init__(self, latent_dim, input_dim):
        super(model, self).__init__()

        if latent_dim > int(input_dim * 0.2):
            latent_dim = int(input_dim * 0.1)
        # encoder
        self.enc1 = nn.Linear(in_features=input_dim, out_features=int(input_dim * 0.8))
        self.norm1 = nn.LayerNorm(int(input_dim * 0.8))
        self.enc2 = nn.Linear(
            in_features=int(input_dim * 0.8), out_features=int(input_dim * 0.6)
        )
        self.norm2 = nn.LayerNorm(int(input_dim * 0.6))
        self.enc3 = nn.Linear(
            in_features=int(input_dim * 0.6), out_features=int(input_dim * 0.4)
        )
        self.norm3 = nn.LayerNorm(int(input_dim * 0.4))
        self.enc4 = nn.Linear(
            in_features=int(input_dim * 0.4), out_features=int(input_dim * 0.2)
        )
        self.norm4 = nn.LayerNorm(int(input_dim * 0.2))
        self.enc5 = nn.Linear(in_features=int(input_dim * 0.2), out_features=latent_dim)
        self.norm5 = nn.LayerNorm(latent_dim)

        # decoder
        self.dec1 = nn.Linear(in_features=latent_dim, out_features=int(input_dim * 0.2))
        self.norm6 = nn.LayerNorm(int(input_dim * 0.2))
        self.dec2 = nn.Linear(
            in_features=int(input_dim * 0.2), out_features=int(input_dim * 0.4)
        )
        self.norm7 = nn.LayerNorm(int(input_dim * 0.4))
        self.dec3 = nn.Linear(
            in_features=int(input_dim * 0.4), out_features=int(input_dim * 0.6)
        )
        self.norm8 = nn.LayerNorm(int(input_dim * 0.6))
        self.dec4 = nn.Linear(
            in_features=int(input_dim * 0.6), out_features=int(input_dim * 0.8)
        )
        self.norm9 = nn.LayerNorm(int(input_dim * 0.8))
        self.dec5 = nn.Linear(in_features=int(input_dim * 0.8), out_features=input_dim)

        # init weights
        torch.nn.init.xavier_uniform(self.enc1.weight)
        torch.nn.init.xavier_uniform(self.enc2.weight)
        torch.nn.init.xavier_uniform(self.enc3.weight)
        torch.nn.init.xavier_uniform(self.enc4.weight)
        torch.nn.init.xavier_uniform(self.enc5.weight)
        torch.nn.init.xavier_uniform(self.dec1.weight)
        torch.nn.init.xavier_uniform(self.dec2.weight)
        torch.nn.init.xavier_uniform(self.dec3.weight)
        torch.nn.init.xavier_uniform(self.dec4.weight)
        torch.nn.init.xavier_uniform(self.dec5.weight)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.norm1(x)
        x = F.relu(self.enc2(x))
        x = self.norm2(x)
        x = F.relu(self.enc3(x))
        x = self.norm3(x)
        x = F.relu(self.enc4(x))
        x = self.norm4(x)
        x = F.relu(self.enc5(x))

        x = F.relu(self.dec1(x))
        x = self.norm6(x)
        x = F.relu(self.dec2(x))
        x = self.norm7(x)
        x = F.relu(self.dec3(x))
        x = self.norm8(x)
        x = F.relu(self.dec4(x))
        x = self.norm9(x)
        x = F.relu(self.dec5(x))

        return x

    def predict(self, x):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        with torch.no_grad():

            x = torch.tensor(x)
            x = x.to(device)

            x = F.relu(self.enc1(x.float()))
            x = self.norm1(x)
            x = F.relu(self.enc2(x))
            x = self.norm2(x)
            x = F.relu(self.enc3(x))
            x = self.norm3(x)
            x = F.relu(self.enc4(x))
            x = self.norm4(x)
            x = F.relu(self.enc5(x))

        return x


class Autoencoder:
    def __init__(self, latent_dim=32, epochs=10, batch_size=1, input_dim=1026):
        self.history = []
        self.model = model(latent_dim, input_dim)
        self.num_epochs = epochs
        self.batch_size = batch_size

    def fit(self, X):

        learning_rate = 1e-1
        # criterion = nn.MSELoss()
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        train_loss = []
        # setting up the device
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        print(device)
        self.model.to(device)

        # todo: implement a data loader class
        data_length = len(X)

        num_batches = data_length // self.batch_size

        for epoch in range(self.num_epochs):

            if self.batch_size > data_length:

                data = X
                d = torch.tensor(data)
                d = torch.autograd.Variable(d)
                d = d.unsqueeze(0)
                d = d.to(device)

                # forward pass
                output = self.model(d.float())
                loss = criterion(output, d.float())

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                for i in range(num_batches):

                    start = i * self.batch_size
                    end = (i + 1) * self.batch_size
                    if i == num_batches - 1:
                        end = data_length

                    data = X[start:end]
                    d = torch.tensor(data)
                    d = torch.autograd.Variable(d)
                    d = d.unsqueeze(0)
                    d = d.to(device)

                    # forward pass
                    output = self.model(d.float())
                    loss = criterion(output, d.float())

                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # log
            loss = loss.item()
            train_loss.append(loss)

            print(
                "Epoch {} of {}, Train Loss: {:.3f}".format(
                    epoch + 1, self.num_epochs, loss
                )
            )
