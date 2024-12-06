import numpy as np
from reversi import reversi
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

# Simple Reversi Neural Network
class ReversiNet(nn.Module):
    def __init__(self):
        super(ReversiNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 64)  # 64 possible moves (8x8 board)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# DQN Agent
class ReversiAgent:
    def __init__(self, lr=0.001, gamma=0.99):
        self.net = ReversiNet()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.gamma = gamma
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.replay_buffer = deque(maxlen=10000)

    def select_action(self, state, valid_moves):
        if np.random.rand() < self.epsilon:
            return random.choice(valid_moves)  # Explore
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        q_values = self.net(state_tensor).detach().numpy().flatten()
        return valid_moves[np.argmax([q_values[m[0]*8+m[1]] for m in valid_moves])]  # Exploit

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        actions = torch.LongTensor(actions)

        # Q-learning update
        q_values = self.net(states)
        next_q_values = self.net(next_states).detach()

        target_q_values = q_values.clone()
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * next_q_values[i].max()

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def get_valid_moves(game, turn):
    valid_moves = []
    for x in range(8):
        for y in range(8):
            if game.step(x, y, turn, commit=False) > 0:
                    valid_moves.append((x, y))
    return valid_moves


# Training Framework
def train_agent(episodes=10000):
    agent = ReversiAgent()
    print("Number of Parameters:", sum(p.numel() for p in agent.net.parameters() if p.requires_grad)) #34305
    for episode in tqdm(range(episodes)):
        game = reversi()
        state = game.board
        done = False
        turn = 1

        while not done:
            valid_moves = get_valid_moves(game, turn)
            if valid_moves:
                action = agent.select_action(state, valid_moves)
                x, y = action
                reward = game.step(x, y, turn)
                if (x,y) in [(0,0),(0,7),(7,0),(7,7)]:
                    reward += 50
                if turn==1:
                    reward += 2*(game.white_count-game.black_count)
                else:
                    reward += 2*(game.black_count-game.white_count)
                turn = 1 if turn==-1 else -1
                next_state = game.board * turn
            else:
                done = True
                next_state = state
                reward = -100
                action = (0, 0)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state

        agent.train()
        #print(f"Episode {episode+1} completed, Epsilon: {agent.epsilon:.2f}")
    torch.save(agent.net.state_dict(), "reversi_dqn_model.pth")

train_agent()