import torch.nn as nn
import torch.nn.functional as F

from spellbond.wordle.env.const import WORDLE_N, MAX_TURNS, ALPHABETS, POSSIBILITIES


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(WORDLE_N * len(POSSIBILITIES) * len(ALPHABETS),
                             WORDLE_N * len(POSSIBILITIES) * len(ALPHABETS) // 2)
        self.fc2 = nn.Linear(WORDLE_N * len(POSSIBILITIES) * len(ALPHABETS) // 2,
                             WORDLE_N * len(POSSIBILITIES) * len(ALPHABETS) // 4)
        self.fc3 = nn.Linear(WORDLE_N * len(POSSIBILITIES) * len(ALPHABETS) // 4,  len(ALPHABETS) * WORDLE_N)
        self.dropout = nn.Dropout(config.model.dropout)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = x.view(-1, len(ALPHABETS), WORDLE_N)
        x = F.softmax(x, dim=1)
        return self.flatten(x)


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(WORDLE_N * len(POSSIBILITIES) * len(ALPHABETS) + MAX_TURNS,
                             WORDLE_N * len(POSSIBILITIES) * len(ALPHABETS) // 2)
        self.fc2 = nn.Linear(WORDLE_N * len(POSSIBILITIES) * len(ALPHABETS) // 2,
                             WORDLE_N * len(POSSIBILITIES) * len(ALPHABETS) // 4)
        self.fc3 = nn.Linear(WORDLE_N * len(POSSIBILITIES) * len(ALPHABETS) // 4, len(ALPHABETS) * WORDLE_N)
        self.fc4 = nn.Linear(len(ALPHABETS) * WORDLE_N, 1)
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc4(x)
