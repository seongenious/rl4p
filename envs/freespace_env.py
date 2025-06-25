import jax
import jax.numpy as jnp
import numpy as np
import math
from typing import NamedTuple, Tuple, List
import random



class FreespaceEnv:
    def __init__(self, dt=0.1, wheelbase=2.5):
        self.dt = dt
        self.wheelbase = wheelbase
        self.goal = None
        self.state = None
        self.max_speed = 10 / 3.6  # 10 kph to m/s
        self.reset()

    def reset(self):
        self.state = State(0.0, 0.0, 0.0, random.uniform(0.0, self.max_speed))
        self.goal = State(
            random.uniform(-30.0, 30.0),
            random.uniform(-30.0, 30.0),
            random.uniform(-math.pi, math.pi),
            0.0
        )
        return self.state, self.goal

    def step(self, action: Action) -> Transition:
        x, y, yaw, v = self.state
        delta, accel = action

        # Kinematic update
        v_next = v + accel * self.dt
        v_next = np.clip(v_next, -self.max_speed, self.max_speed)

        x_next = x + v_next * math.cos(yaw) * self.dt
        y_next = y + v_next * math.sin(yaw) * self.dt
        yaw_next = yaw + v_next / self.wheelbase * math.tan(delta) * self.dt
        yaw_next = (yaw_next + math.pi) % (2 * math.pi) - math.pi

        next_state = State(x_next, y_next, yaw_next, v_next)

        reward = self._compute_reward(next_state)
        done = self._check_done(next_state)

        self.state = next_state
        return Transition(State(x, y, yaw, v), action, next_state, reward, done)

    def _compute_reward(self, state: State) -> float:
        path_length = self._reeds_shepp_path_length(state, self.goal)
        return -path_length  # penalize distance

    def _check_done(self, state: State) -> bool:
        dx = state.x - self.goal.x
        dy = state.y - self.goal.y
        dyaw = abs((state.yaw - self.goal.yaw + math.pi) % (2 * math.pi) - math.pi)
        distance = math.sqrt(dx ** 2 + dy ** 2)
        return distance < 0.1 and dyaw < math.radians(1.0)

    def _reeds_shepp_path_length(self, start: State, goal: State) -> float:
        
        # Dummy implementation (placeholder for real reeds-shepp planner)
        dx = goal.x - start.x
        dy = goal.y - start.y
        dyaw = abs((goal.yaw - start.yaw + math.pi) % (2 * math.pi) - math.pi)
        turning_penalty = 1.0
        return math.sqrt(dx ** 2 + dy ** 2) + turning_penalty * dyaw

    def sample_transitions(self, num_samples: int) -> List[Transition]:
        transitions = []
        for _ in range(num_samples):
            action = Action(
                delta=random.uniform(-0.5, 0.5),
                accel=random.uniform(-2.0, 2.0)
            )
            transitions.append(self.step(action))
        return transitions


# 테스트 및 예시 출력
env = FreespaceEnv()
samples = env.sample_transitions(5)

# import pandas as pd
# import ace_tools as tools 
# tools.display_dataframe_to_user(name="Sampled Transitions", dataframe=pd.DataFrame(samples))
