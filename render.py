import matplotlib.pyplot as plt
from SnakeGame.snakeGame import SnakeGame
from stable_baselines3 import PPO

env = SnakeGame(30, 30, 'PPO_2/1340000.zip')
observation = env.reset()

model = PPO.load(f"models/PPO_3/340000.zip")

fig, ax = plt.subplots()
grid = env.board.board
cax = ax.imshow(grid, cmap='viridis')
fig.colorbar(cax)
plt.show(block=False)
plt.draw()
plt.pause(1)

while True:
    # try:
    #     action = int(input("Enter action: "))
    # except:
    #     action = 0
    action = model.predict(observation)[0]
    observation, _, done = env.step(action)
    if done:
        observation = env.reset()
    grid = env.board.board
    cax.set_data(grid)
    plt.draw()
    plt.pause(0.1)
