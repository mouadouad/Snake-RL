# Snake-RL
Reinforcement learning models to play snake
python 3.9
venv\Scripts\activate
pip install -r requirements.txt
tensorboard --logdir logs

observations tried:
vector of distance from obstacle (best)
a grid of a smaller resolution where the head of the snake is in the middle (second best)
the whole board (worst)
the board and a vector of the position of the head
the board but adapted to the distance to the head