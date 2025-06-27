import reeds_shepp as rs
import matplotlib.pyplot as plt
import numpy as np


def main():
    q0 = (0.05, 0, np.pi/8)
    q1 = (0, 0, 0)
    r = 6
    ds = 0.1
    points = rs.path_sample(q0, q1, r, step_size=ds)
    for i, point in enumerate(points):
        print(f'[{i}] {point}')


    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    
    path = np.array(points)  # shape: (N, 5)
    pos = path[:, :2]         # (x, y)
    yaw = path[:, 2]          # yaw

    delta = pos[1:] - pos[:-1]  # displacement vector
    heading = np.stack([np.cos(yaw[:-1]), np.sin(yaw[:-1])], axis=-1)

    dot = np.sum(delta * heading, axis=-1)
    direction = np.sign(dot)
    print(direction)  # shape: (N-1,), values: +1 or -1
    
    s = points[:, 3]
    signs = np.sign(s)
    change = np.where(signs[1:] != signs[:-1])
    print(change)

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, label="RS Path", linewidth=2)

    plt.plot(q0[0], q0[1], 'go', label="Start")
    plt.arrow(q0[0], q0[1], 0.5*np.cos(q0[2]), 0.5*np.sin(q0[2]),
        head_width=0.2, color='g')

    plt.plot(q1[0], q1[1], 'ro', label="Goal")
    plt.arrow(q1[0], q1[1], 0.5*np.cos(q1[2]), 0.5*np.sin(q1[2]),
        head_width=0.2, color='r')

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.savefig('./test.png')

if __name__ == "__main__":
    main()
