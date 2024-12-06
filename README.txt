Reversi AI

Implementation of Greedy; MinMax algorithm with alpha-beta pruning and limited depth of 4; and Deep-Q Network with epsilon-greedy exploration.
Deep-Q Network has 34,305 parameters and was trained for 10,000 epochs. File 'reversi_dqn_model.pth' contains model weights for Deep-Q Network.

Execution:
Run following commands in different terminals.
For Min-Max vs greedy
Step 1: python reversi_server.py
Step 2: python greedy_player.py
Step 3: python minmax_player.py

For DQN vs greedy
Step 1: python reversi_server.py
Step 2: python greedy_player.py
Step 3: python DQN_player.py

Swap step 2 and 3 to change color of players (who goes first)
Note: make sure you are useing environment with Python 3.11 or higher.
