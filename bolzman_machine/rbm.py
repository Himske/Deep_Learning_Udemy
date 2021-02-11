import numpy as np
import pandas as pd
import torch

# importing the dataset
movies = pd.read_csv('bolzman_machine/ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('bolzman_machine/ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('bolzman_machine/ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# preparing training and test sets
training_set = pd.read_csv('bolzman_machine/ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int64')

test_set = pd.read_csv('bolzman_machine/ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int64')

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


training_set = convert(data=training_set, nb_users=nb_users, nb_movies=nb_movies)
test_set = convert(data=test_set, nb_users=nb_users, nb_movies=nb_movies)

# convert datasets into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# convert the ratings into binary ratings 1 (liked, ratings 3-5) or 0 (not liked, ratings 1-2)
# unrated movies will get a rating of -1
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


class RBM():
    def __init__(self, num_vis, num_hid):
        self.weights = torch.randn(num_hid, num_vis)
        self.bias_hid = torch.randn(1, num_hid)
        self.bias_vis = torch.randn(1, num_vis)

    def sample_hidden(self, visible_nodes):
        weigths_neurons = torch.mm(visible_nodes, self.weights.t())
        activation = weigths_neurons + self.bias_hid.expand_as(weigths_neurons)
        probability_of_hidden_given_visible = torch.sigmoid(activation)
        return probability_of_hidden_given_visible, torch.bernoulli(probability_of_hidden_given_visible)

    def sample_visible(self, hidden_nodes):
        weigths_neurons = torch.mm(hidden_nodes, self.weights)
        activation = weigths_neurons + self.bias_vis.expand_as(weigths_neurons)
        probability_of_visible_given_hidden = torch.sigmoid(activation)
        return probability_of_visible_given_hidden, torch.bernoulli(probability_of_visible_given_hidden)

    def train(self, vis_nodes_at_0, vis_nodes_after_k_sampl, prob_hid_at_0, prob_hid_at_k_sampl):
        self.weights += (torch.mm(vis_nodes_at_0.t(), prob_hid_at_0) -
                         torch.mm(vis_nodes_after_k_sampl.t(), prob_hid_at_k_sampl)).t()
        self.bias_vis += torch.sum((vis_nodes_at_0 - vis_nodes_after_k_sampl), 0)
        self.bias_hid += torch.sum((prob_hid_at_0 - prob_hid_at_k_sampl), 0)

    def predict(self, vis_nodes):
        _, h = self.sample_hidden(vis_nodes)
        _, v = self.sample_visible(h)
        return v


num_vis = len(training_set[0])
num_hid = 100  # number of features we want to detect
batch_size = 100

rbm = RBM(num_vis=num_vis, num_hid=num_hid)

# train RBM
epochs = 10

for epoch in range(1, epochs + 1):
    train_loss = 0
    s = 0.

    for id_user in range(0, nb_users - batch_size, batch_size):
        v0 = training_set[id_user:id_user + batch_size]
        vk = training_set[id_user:id_user + batch_size]
        ph0, _ = rbm.sample_hidden(v0)

        for k in range(10):
            _, hk = rbm.sample_hidden(vk)
            _, vk = rbm.sample_visible(hk)
            vk[v0 < 0] = v0[v0 < 0]  # unrated movies

        phk, _ = rbm.sample_hidden(vk)
        rbm.train(vis_nodes_at_0=v0, vis_nodes_after_k_sampl=vk, prob_hid_at_0=ph0, prob_hid_at_k_sampl=phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.

    print(f'Epoch: {epoch} loss: {train_loss / s}')

# testing RBM
test_loss = 0
s = 0.

for id_user in range(nb_users):
    vt = test_set[id_user:id_user + 1]

    if len(vt[vt >= 0]) > 0:
        prediction = rbm.predict(vt)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - prediction[vt >= 0]))
        s += 1.

print(f'Test loss: {test_loss / s}')
