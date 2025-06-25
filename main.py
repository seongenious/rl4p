import matplotlib.pyplot as plt 
import numpy as np

from envs.datatypes import State, Action, Transition
from envs.reeds_shepp import rs_path

start = State(x=0, y=0, yaw=0, v=0)
goal = State(x=0, y=1, yaw=0.1, v=0)

l, points = rs_path(start, goal)
points = np.array(points)
x, y = points[:, 0], points[:, 1]

plt.figure(figsize=(6, 6))
plt.plot(x, y, label="rs path", linewidth=1)

plt.plot(start.x, start.y, 'go', label="start")
plt.arrow(start.x, start.y, 0.5 * np.cos(start.yaw), 0.5 * np.sin(start.yaw),
    head_width=0.2, color='g')

plt.plot(goal.x, goal.y, 'ro', label="goal")
plt.arrow(goal.x, goal.y, 0.5 * np.cos(goal.yaw), 0.5 * np.sin(goal.yaw),
    head_width=0.2, color='r')

plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title(f"Reeds-Shepp Path: {l} m")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.savefig('rspath.png')
