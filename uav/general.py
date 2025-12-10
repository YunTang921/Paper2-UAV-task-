"""
Purpose: visualize all "general" series data in Data/ using every file and field.
Outputs:
  - Cost line plot for each file
  - Cost heatmap for 2D cost arrays
  - Time curve (with std band if available) for files that contain time/time_std
Images are saved to new results/General.
"""
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path("Data")
OUTPUT_DIR = Path("new results") / "General"

# Explicit list to ensure every general-related file is used
GENERAL_FILES = [
    "Beta_Gneral_10000_777.pkl",
    "General_Final_10000_777.pkl",
    "General_New_10000_777.pkl",
    "General_New_100_777.pkl",
    "Genral_New_100_777.pkl",
    "Gneral_10000_777.pkl",
    "Gneral_100_777.pkl",
    "Gneral_Ablation_10000_777.pkl",
    "Gneral_New_10000_777.pkl",
]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def ensure_output():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_cost_lines(label: str, cost_array: np.ndarray):
    # cost_array can be 1D or 2D; ensure 2D for unified handling
    if cost_array.ndim == 1:
        cost_array = cost_array[None, :]
    epochs = np.arange(1, cost_array.shape[1] + 1)
    plt.figure(figsize=(8, 5))
    for i, row in enumerate(cost_array):
        plt.plot(epochs, row, marker="o", linewidth=2, label=f"series {i+1}")
    plt.xlabel("Task index")
    plt.ylabel("Cost")
    plt.title(f"Cost curves - {label}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    out_path = OUTPUT_DIR / f"{label}_cost_lines.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_cost_heatmap(label: str, cost_array: np.ndarray):
    if cost_array.ndim < 2:
        return
    plt.figure(figsize=(8, 4))
    im = plt.imshow(cost_array, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="Cost")
    plt.xlabel("Task index")
    plt.ylabel("Series index")
    plt.title(f"Cost heatmap - {label}")
    out_path = OUTPUT_DIR / f"{label}_cost_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def plot_time_curve(label: str, time_list, time_std):
    x = np.arange(1, len(time_list) + 1)
    time_arr = np.array(time_list, dtype=float)
    plt.figure(figsize=(8, 4))
    plt.plot(x, time_arr, marker="s", linewidth=2, color="tab:orange", label="Time")
    if time_std is not None:
        std = float(time_std)
        plt.fill_between(x, time_arr - std, time_arr + std, color="tab:orange", alpha=0.2, label="± std")
    plt.xlabel("Task index")
    plt.ylabel("Time")
    plt.title(f"Time curve - {label}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    out_path = OUTPUT_DIR / f"{label}_time.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def main():
    ensure_output()

    for filename in GENERAL_FILES:
        path = DATA_DIR / filename
        if not path.exists():
            print(f"Skip missing {path}")
            continue
        payload = load_pickle(path)
        label = filename.replace(".pkl", "")

        cost = payload.get("cost")
        if cost is None:
            continue

        cost_arr = np.array(cost, dtype=float)
        plot_cost_lines(label, cost_arr)
        plot_cost_heatmap(label, cost_arr)

        if "time" in payload:
            time_list = payload["time"]
            time_std = payload.get("time_std", None)
            plot_time_curve(label, time_list, time_std)


if __name__ == "__main__":
    main()
