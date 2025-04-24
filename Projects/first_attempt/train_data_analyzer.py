import os
import re
import matplotlib.pyplot as plt

def extract_values_from_filename(filename):
    # Updated to support float values in rew (positive or negative, with decimals)
    match = re.search(r'ep_(\d+)_rew_(-?\d+(?:\.\d+)?)', filename)
    if match:
        x = int(match.group(1))
        y = float(match.group(2))
        return x, y
    return None

def process_folder(folder_path):
    data = []

    for filename in os.listdir(folder_path):
        result = extract_values_from_filename(filename)
        if result:
            data.append(result)

    # Sort by ep (x)
    data.sort(key=lambda pair: pair[0])
    x_values, y_values = zip(*data) if data else ([], [])
    return x_values, y_values

def print_min_max(x, y):
    min_index = y.index(min(y))
    max_index = y.index(max(y))
    print(f"Min rew: {y[min_index]} at ep: {x[min_index]}")
    print(f"Max rew: {y[max_index]} at ep: {x[max_index]}")

def plot_graph(x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')
    plt.title('Graph of Rewards (Rew) vs Episodes (EP)')
    plt.xlabel('EP')
    plt.ylabel('Rew')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    folder = r"C:\Users\mcpek\IsaacLab\logs\rl_games\car_driver\2025-04-22_18-13-08\nn"
    x_vals, y_vals = process_folder(folder)
    if x_vals and y_vals:
        print_min_max(x_vals, y_vals)
        plot_graph(x_vals, y_vals)
    else:
        print("No matching files found.")
