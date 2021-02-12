import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.modules.loss import MSELoss

# importing the dataset
# uncomment if they are going to be used
movies = pd.read_csv('dataset/ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1').to_numpy()
# users = pd.read_csv('dataset/ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
# ratings = pd.read_csv('dataset/ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# preparing training and test sets
training_set = pd.read_csv('dataset/ml-100k/u1.base', delimiter='\t').to_numpy()
test_set = pd.read_csv('dataset/ml-100k/u1.test', delimiter='\t').to_numpy()

combined = np.append(training_set, test_set, axis=0)

# getting the max number of users and movies
nb_users = int(max(combined[:, 0]))
nb_movies = int(max(combined[:, 1]))


# convert the data into an array of arrays of user ratings setting 0 for non rated movies
def convert(data, nb_users=0, nb_movies=0):
    new_data = list()
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        user_ratings = np.zeros(nb_movies, dtype=int)
        user_ratings[id_movies - 1] = id_ratings
        new_data.append(list(user_ratings))
    return new_data


def convert_better(data):
    new_data = list()
    nb_movies = int(max(movies[:, 0]))

    for user_id in np.unique(data[:, 0]):
        id_movies = data[:, 1][data[:, 0] == user_id]
        id_ratings = data[:, 2][data[:, 0] == user_id]
        user_ratings = np.zeros(nb_movies, dtype=int)
        user_ratings[id_movies - 1] = id_ratings
        new_data.append(list(user_ratings))
    return new_data


training_set_converted = convert_better(training_set)
test_set_converted = convert_better(test_set)

training_set = convert(data=training_set, nb_users=nb_users, nb_movies=nb_movies)
test_set = convert(data=test_set, nb_users=nb_users, nb_movies=nb_movies)

# convert datasets into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
training_set_converted = torch.FloatTensor(training_set_converted)
test_set_converted = torch.FloatTensor(test_set_converted)


# create the architecture of the neural network
class SAE(nn.Module):
    def __init__(self, nb_io_nodes):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_io_nodes, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_io_nodes)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


sae = SAE(nb_movies)
criterion = MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# training the SAE
# epochs = 200
for epoch in range(1, 201):  # epochs = 200
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae.forward(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.
            optimizer.step()
    print(f'epoch: {epoch} loss: {train_loss / s}')

# testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output = sae.forward(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.item() * mean_corrector)
        s += 1.
print(f'Test loss: {test_loss / s}')
