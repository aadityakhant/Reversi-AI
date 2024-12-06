#Zijie Zhang, Sep.24/2023

import numpy as np
import socket, pickle
from reversi import reversi
import torch
import torch.nn as nn

# Define the DQN model
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

def get_valid_moves(game, turn):
    valid_moves = []
    for x in range(8):
        for y in range(8):
            if game.step(x, y, turn, commit=False) >= 0:
                    valid_moves.append((x, y))
    return valid_moves

def main():
    model = ReversiNet()
    model.load_state_dict(torch.load("reversi_dqn_model.pth", weights_only=True))
    model.eval()
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:
        #Receive play request from the server
        #turn : 1 --> you are playing as white | -1 --> you are playing as black
        #board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)
        game.board = board
        #Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return
        
        #Debug info
        valid_moves = get_valid_moves(game, turn)
        if valid_moves:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(game.board*turn).unsqueeze(0).unsqueeze(0)
                q_values = model(state_tensor).detach().numpy().flatten()
                x, y = valid_moves[np.argmax([q_values[m[0]*8+m[1]] for m in valid_moves])]
        else:
            x, y = -1, -1
        #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x,y]))
        
if __name__ == '__main__':
    main()