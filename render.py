import matplotlib.pyplot as plt
from SnakeGame.snakeGame import SnakeGame
from stable_baselines3 import PPO

# PLAYER2_MODEL = "PPO_16/340000"
PLAYER2_MODEL = None
MODEL_TO_LOAD = "PPO_25/4560000"

model = PPO.load(f"models/{MODEL_TO_LOAD}.zip")

# env = SnakeGame(40, 40, "PPO_1/940000.zip")
env = SnakeGame(40, 9, PLAYER2_MODEL)


observation, info = env.reset()
print(observation)

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
    #     action =0
    # observation['image'] = observation['image'].reshape((1, 42, 29))  # Adding channel dimension for a grayscale observation

    action = model.predict(observation)[0]
    observation, _, done, info = env.step(action)
    if done:
        observation, info = env.reset()
    grid = env.board.board
    cax.set_data(grid)
    plt.draw()
    plt.pause(0.1)
