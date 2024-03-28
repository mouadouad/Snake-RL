import matplotlib.pyplot as plt
from SnakeGame.snakeGame import SnakeGame

env = SnakeGame(30, 15, 'PPO_3/1340000.zip')
env.reset()

fig, ax = plt.subplots()
grid = env.board.board
cax = ax.imshow(grid, cmap='viridis')
fig.colorbar(cax)
plt.show(block=False)
plt.draw()
plt.pause(1)

while True:
    try:
        action = int(input("Enter action: "))
    except:
        action = 0
    env.step(action)
    grid = env.board.board
    cax.set_data(grid)
    plt.draw()
    plt.pause(0.1)
