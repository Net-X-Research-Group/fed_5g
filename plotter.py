import numpy as np
import matplotlib.pyplot as plt


def plot_metrics(args, loss_plot: list, accuracy_plot: list, perdevice_training_time_plot: list, training_time_avg_plot: list, perdevice_fit_time_plot: list, fit_time_avg_plot: list):
    """Plot all tracked metrics."""
    rounds = range(1, len(accuracy_plot) + 1)

    # Create a figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot accuracy
    ax1.plot(rounds, accuracy_plot, marker='o')
    ax1.set_xlabel('Rounds')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy vs. Rounds')
    ax1.grid(True)
    # Plot loss
    ax2.plot(rounds, loss_plot, marker='o', color='red', label='Loss (Dist.)')
    # ax2.plot(rounds, eval_losses_plot, marker='o', color='blue', label='Loss (Eval)')
    ax2.set_xlabel('Rounds')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs. Rounds')
    ax2.legend()
    ax2.grid(True)

    # Find the maximum length among all rounds
    max_num_devices = max(len(round_times) for round_times in perdevice_training_time_plot)

    # Pad each round's data with zeros to match max length
    padded_times = [round_times + [0] * (max_num_devices - len(round_times))
                    for round_times in perdevice_training_time_plot]

    # Reorganize data to group by device
    device_times = []
    for device_idx in range(max_num_devices):
        device_times.append([round_times[device_idx] for round_times in padded_times])

    # Create a color map for different devices
    colors = plt.cm.rainbow(np.linspace(0, 1, max_num_devices))

    # Plot a line for each device
    for device_idx, times in enumerate(device_times):
        ax3.plot(rounds, times, marker='o', color=colors[device_idx], linestyle='--',
                 label=f'Device {device_idx + 1}')
    ax3.plot(rounds, training_time_avg_plot, marker='^', color='red', label='Average')
    ax3.set_xlabel('Rounds')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Computation Time per Device per Round')
    ax3.grid(True)
    ax3.legend()

    # Plot communication time
    # Find the maximum length among all rounds
    max_num_devices = max(len(round_times) for round_times in perdevice_fit_time_plot)

    # Pad each round's data with zeros to match max length
    padded_times = [round_times + [0] * (max_num_devices - len(round_times))
                    for round_times in perdevice_fit_time_plot]

    # Reorganize data to group by device
    device_times_fit = []
    for device_idx in range(max_num_devices):
        device_times_fit.append([round_times[device_idx] for round_times in padded_times])

    # Create a color map for different devices
    colors = plt.cm.rainbow(np.linspace(0, 1, max_num_devices))

    # Plot a line for each device
    for device_idx, times in enumerate(device_times_fit):
        ax4.plot(rounds, times, marker='o', color=colors[device_idx], linestyle='--',
                 label=f'Device {device_idx + 1}')
    ax4.plot(rounds, fit_time_avg_plot, marker='^', color='purple', label='Avg. Fit Time')
    ax4.set_xlabel('Rounds')
    ax4.set_ylabel('Time (seconds)')
    # ax4.set_title('Average Communication Time per Round')
    ax4.set_title('Communication Cost vs. Rounds')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(f'FedAvg_{args.min_num_clients}C_{args.rounds}R_Metrics.png')
    plt.show()