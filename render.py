import matplotlib.pyplot as plt
from SnakeGame.snakeGame import SnakeGame
from stable_baselines3 import PPO


model = PPO.load(f"models/PPO_40/3940000.zip")

env = SnakeGame(40, 40, "PPO_40/3940000")
observation, info = env.reset()

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
    observation, _, done, info = env.step(action)
    if done:
        observation, info = env.reset()
    grid = env.board.board
    cax.set_data(grid)
    plt.draw()
    plt.pause(0.1)
