import gymnasium as gym
import numpy as np
from gymnasium import spaces

GRID_SIZE = 10
TARGET_STATE = 7
INITIAL_STATE = 2


def integer_to_one_hot(integer_value, n=GRID_SIZE-1):

    if integer_value < 0 or integer_value > n:
        raise ValueError("Integer value is out of range [0, n]")

    one_hot_vector = np.zeros(n + 1)
    one_hot_vector[integer_value] = 1
    return one_hot_vector


def one_hot_to_integer(one_hot_vector):
    if not isinstance(one_hot_vector, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    if len(one_hot_vector.shape) != 1:
        raise ValueError("Input must be a 1-dimensional array")

    return np.argmax(one_hot_vector)

# ToDo: Add to custom environment the possibility to jump more than one side within the grid. 
#  This could be intrested to study out of distribution actions 
#  (e.g. generate only actions that are one-step and in the offline training phase see if some algorithms create 
#  new actions. (for instance check multi-step actions vs out of distrib. param (tau in bcq))


class CustomGridEnvGymnasium(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, grid_size:int,  max_episode_steps=20, discrete_action=True,
                 render_mode="human"):

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        initial_state = 3
        target_state = 7


        self.grid_size = grid_size
        self.target_state = integer_to_one_hot(target_state)
        self.discrete_action = discrete_action

        if self.discrete_action:
            self.action_space = spaces.Discrete(2)  # Two possible actions: move left or move right
        else:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64)

        self.observation_space = spaces.Box(low=0., high=1.0, shape=(grid_size,), dtype=np.float64)#spaces.MultiDiscrete([2]*(grid_size))#spaces.Discrete(grid_size)

        self.initial_state = integer_to_one_hot(initial_state)
        self.current_state = self.initial_state

        self.steps = 0  # Variable to keep track of the number of steps
        self.max_episode_steps = max_episode_steps

        # Two args. below used in order to print the sequence of moves within an episode
        self.move_sequence = []
        self.is_done = False
        self.is_truncated = False
        self.episode_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self.initial_state
        self.steps = 0  # Reset the step count
        self.is_done = False
        self.is_truncated = False
        self.move_sequence = []
        self.episode_reward = 0.0
        return self.current_state, {}

    def step(self, action):

        if not self.discrete_action:
            action = round(action[0])

        if action == 0:  # Move left
            self.current_state = integer_to_one_hot(max(one_hot_to_integer(self.current_state) - 1, 0))
        elif action == 1:  # Move right
            self.current_state = integer_to_one_hot(min(one_hot_to_integer(self.current_state) + 1, self.grid_size - 1))


        self.steps += 1  # Increment the step count

        done = one_hot_to_integer(self.current_state) == one_hot_to_integer(self.target_state)
               #or self.steps >= self.max_episode_steps  # Terminate if target is reached or max steps exceeded
        self.move_sequence.append(action)

        truncated = False
        if self.steps == self.max_episode_steps:
            truncated = True
            self.is_truncated = True


        reward = 1.0
        if done:
            reward += 0.0
            self.is_done = True
            info = {"move_sequence": str(self.move_sequence)}
        else:
            current_state_int = one_hot_to_integer(self.current_state)
            target_state_int = one_hot_to_integer(self.target_state)
            max_dist = self.grid_size
            #rew_1 = 1.0  - abs((current_state_int - target_state_int)/(1.0*max_dist))
            reward = - 0.5*(self.steps/self.max_episode_steps +
                            abs((current_state_int - target_state_int)/(1.0*max_dist)))

            info = {"move_sequence": ""}

        self.episode_reward += reward
        #print("rewww ",reward)

        return self.current_state, reward, done, truncated, info

    def render_with_policy_values(self, policy_probs):
        policy_values = policy_probs
        grid = ['-'] * self.grid_size
        grid[one_hot_to_integer(self.current_state)] = 'S'  # Mark the moving state with 'S'
        grid[one_hot_to_integer(self.target_state)] = 'X'   # Mark the target state with 'X'

        if policy_values:
            values_str = " | ".join(f"{value:.2f}" for value in policy_values)
            print(f"States: {' '.join(grid)}")
            print(f"Values: {values_str}")
        else:
            print(' '.join(grid))

        if self.is_done or self.is_truncated:
            print(f"###### {self.move_sequence} ######")

    def render(self):
        if self.render_mode == "human":
            #policy_values = self.move_sequence[-1]#kwargs["policy_values"]

            grid = ['-'] * self.grid_size
            grid[one_hot_to_integer(self.current_state)] = 'S'  # Mark the moving state with 'S'
            grid[one_hot_to_integer(self.target_state)] = 'X'   # Mark the target state with 'X'
            print(' '.join(grid))

            if self.is_done or self.is_truncated:
                print(f"###### {self.move_sequence} -- episode_rew: {self.episode_reward} ###### ")
        else:
            raise Warning("The only render mode support so far is 'human' ")

    def set_initial_state(self, initial_state):
        self.initial_state = integer_to_one_hot(initial_state)

    def set_target_state(self, target_state):
        self.target_state = integer_to_one_hot(target_state)


