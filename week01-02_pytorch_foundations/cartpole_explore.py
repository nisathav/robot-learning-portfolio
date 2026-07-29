import gymnasium as gym

"""
1. What are the 4 values in CartPole's observation space, in order?
    - Carts position
    - Carts velocity
    - Pole's angular position
    - Pole's angular velocity

2. How many possible actions are there, and what do they represent?
    - 0, cart moving to left side
    - 1, cart moving to right side

3. What exactly triggers the reward, and what's the actual value given per step?
    - every action per timestamp triggers the reward, +1 given per step regardless of the cart's motion
"""

def run_episode(seed=None):
    env = gym.make("CartPole-v1")

    observation, info = env.reset(seed=seed)

    total_reward = 0
    step_count = 0

    while True:
        # Random action
        action = env.action_space.sample() # it generates uniform random actions, CartPole has two discrete actions those are 0 (move left) and 1 (move right)

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    env.close()

    return step_count, total_reward


# Run multiple episodes
num_episodes = 5

for episode in range(num_episodes):
    # Use different seeds (or set seed=None for completely random episodes)
    steps, total_reward = run_episode(seed=episode)

    print(
        f"Episode {episode + 1}: "
        f"Steps Survived = {steps}, "
        f"Total Reward = {total_reward}"
    )