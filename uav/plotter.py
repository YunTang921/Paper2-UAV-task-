# 作用: 将任务点与解序列可视化为路径图（单机版，可选额外保存副本到指定目录）；依赖: 无其它项目文件; 被依赖: 手动分析/展示脚本
# 提示: 这是函数库，需传入 mission 和 solution 调用 plot_result 才会出图
import matplotlib.pyplot as plt
from pathlib import Path
import time
import numpy as np
import math


def contact_point(point, center, radius):
    assert isinstance(point, tuple), "point must be tuple"
    assert isinstance(center, tuple), "center must be tuple"

    point = np.array(point)
    center = np.array(center)

    d = np.linalg.norm(point - center)

    if d <= radius:
        vector = (center - point) / d
        heading = - (radius - d) * vector
        contact_point = point + heading

    else:
        h = abs(center[1] - point[1])
        l = (d ** 2 - radius ** 2) ** 0.5
        theta = math.asin(h / d)
        beta = math.acos(l / d)
        alpha = theta - beta

        dx = l * math.cos(alpha)
        dy = l * math.sin(alpha)

        x_sign = 1 if center[0] > point[0] else -1
        y_sign = 1 if center[1] > point[1] else -1

        contact_point = point + np.array((x_sign * dx, y_sign * dy))

    return contact_point


###########################################################
# Plot
###########################################################

def plot_result(mission, solution, args, save_dir=None, auto_axis=False):
    assert isinstance(mission, np.ndarray), "The mission must be numpy array"
    assert isinstance(solution, np.ndarray), "The solution must be numpy array"

    Depot = mission[:1, :]
    Area = mission[1:args.coverage_num + 1, :]
    Visit = mission[args.coverage_num + 1:, :]

    prev = Depot[0]

    fit, ax = plt.subplots(1, 2)
    ax[0].set_title(
        "%d visiting, %d coverage, %d pick_place" % (args.visiting_num, args.coverage_num, args.pick_place_num))
    ax[1].set_title("Solution")

    ax[0].scatter(prev[:1], prev[1:2], marker='s', c='k', s=30, label='Depot')
    ax[1].scatter(prev[:1], prev[1:2], marker='s', c='k', s=30, label='Depot')

    theta = np.radians(np.linspace(0, 360 * 5, 1000))
    just_returned = False
    for idx, i in enumerate(solution[1:], start=1):
        # handle depot return mid-route
        if i == 0:
            # mid-returns use blue dash, final return uses black dash
            is_last = (idx == len(solution) - 1)
            color = 'k--' if is_last else 'b--'
            ax[1].plot([prev[0], Depot[0][0]], [prev[1], Depot[0][1]], color, linewidth=0.5)
            prev = Depot[0]
            just_returned = True
            continue

        task = mission[i]

        if task[-2] == 1:
            point = task[:2]
            ax[0].scatter(point[:1], point[1:2], marker='s', color='b', s=10, label='Visiting')
            ax[1].scatter(point[:1], point[1:2], marker='s', color='b', s=10, label='Visiting')
            ax[1].plot([prev[0], point[0]], [prev[1], point[1]], 'r-', linewidth=0.5)
            prev = point
            just_returned = False

        elif task[-1] == 1:
            x, y, r = task[:3]
            ax[0].add_patch(plt.Circle((x, y), r, fill=False))
            ax[1].add_patch(plt.Circle((x, y), r, fill=False))

            spiral_r = theta / 31 * r
            spiral_x = spiral_r * np.cos(theta) + x
            spiral_y = spiral_r * np.sin(theta) + y
            ax[1].plot(spiral_x, spiral_y, 'r-', linewidth=0.5)

            contact = contact_point((prev[0], prev[1]), (x, y), r)
            ax[1].plot([prev[0], contact[0]], [prev[1], contact[1]], 'r-', linewidth=0.5)
            prev = np.array([x, y])
            just_returned = False

        elif task[-3] == 1:
            pick_point = task[:2]
            place_point = task[3:5]
            points = np.concatenate((pick_point[None, :], place_point[None, :]), axis=0)

            ax[1].plot([prev[0], pick_point[0]], [prev[1], pick_point[1]], 'r-', linewidth=0.5)
            ax[0].scatter(points[:, 0], points[:, 1], marker='D', color='m', s=20)
            ax[1].scatter(points[:, 0], points[:, 1], marker='D', color='m', s=20)
            ax[0].arrow(pick_point[0], pick_point[1], 0.8 * (place_point[0] - pick_point[0]),
                        0.8 * (place_point[1] - pick_point[1]), width=0.001, color='c', head_width=0.006)
            ax[1].arrow(pick_point[0], pick_point[1], 0.8 * (place_point[0] - pick_point[0]),
                        0.8 * (place_point[1] - pick_point[1]), width=0.001, color='c', head_width=0.006)
            prev = place_point
            just_returned = False

    # final return to depot (use depot coords, not (0,0))
    ax[1].plot([prev[0], Depot[0][0]], [prev[1], Depot[0][1]], 'k--', linewidth=0.5, label='Last path')

    if auto_axis:
        pad = 0.05
        x_min, y_min = mission[:, 0].min(), mission[:, 1].min()
        x_max, y_max = mission[:, 0].max(), mission[:, 1].max()
        ax_limit = (x_min - pad, x_max + pad, y_min - pad, y_max + pad)
    else:
        ax_limit = (-0.05, 1.05, -0.05, 1.05)

    ax[0].set_xlim((ax_limit[0], ax_limit[1]))
    ax[0].set_ylim((ax_limit[2], ax_limit[3]))
    ax[0].set_aspect('equal')

    ax[1].set_xlim((ax_limit[0], ax_limit[1]))
    ax[1].set_ylim((ax_limit[2], ax_limit[3]))
    ax[1].set_aspect('equal')

    plt.show()

    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = Path(save_dir) / f"plot_result_{ts}.png"
        fit.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"Saved plot to {out_path}")
