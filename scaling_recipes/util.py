import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from zuko.utils import odeint

import os
import seaborn as sns  
from matplotlib.colors import LogNorm

def sweep_plot(logs_df, cfg, show=False):
    # Apply smoothing to the evaluation loss for smoother curves
    logs_df = logs_df.sort_values(['width', 'log2lr'])
    logs_df['eval_loss_smooth'] = logs_df.groupby('width')['eval_loss'].transform(
        lambda x: x.rolling(window=5, min_periods=1, center=True).mean()
    )
    # Use the smoothed values for plotting, but keep the original for reference
    logs_df['eval_loss_original'] = logs_df['eval_loss']
    logs_df['eval_loss'] = logs_df['eval_loss_smooth']

    # Apply smoothing to the evaluation accuracy for smoother curves, similar to loss
    logs_df['eval_accuracy_smooth'] = logs_df.groupby('width')['eval_accuracy'].transform(
        lambda x: x.rolling(window=5, min_periods=1, center=True).mean()
    )
    # Use the smoothed values for plotting, but keep the original for reference
    logs_df['eval_accuracy_original'] = logs_df['eval_accuracy']
    logs_df['eval_accuracy'] = logs_df['eval_accuracy_smooth']
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    # Define a color palette - using a sequential color map for width progression
    palette = sns.color_palette("viridis", n_colors=len(logs_df['width'].unique()))
    
    # First subplot for eval_loss
    sns.lineplot(x='log2lr', y='eval_loss', hue='width', data=logs_df[(logs_df['model_type']=='MLP')&
                                                                   (logs_df['eval_loss']>0)&
                                                                   (logs_df['epoch']==0)], 
                palette=palette, ax=ax1)
    # Set y-axis to log scale for the loss plot
    ax1.set_yscale('log')
    ax1.set_xscale('log')

    ax1.set_title('Evaluation Loss')
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(0, 2)
    # Second subplot for eval_accuracy
    sns.lineplot(x='log2lr', y='eval_accuracy', hue='width', data=logs_df[(logs_df['model_type']=='MLP')&
                                                                   (logs_df['epoch']==0)], 
                palette=palette, ax=ax2)
    ax2.set_xscale('log')
    ax2.set_title('Evaluation Accuracy')
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(80, 100)
    
    # Add legends with better formatting
    #ax1.legend(title='Width', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Remove the default legend from the first subplot
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    
    # Create a custom colorbar for width with log scale
    # Only use powers of 2 for the colorbar
    max_width = max(logs_df['width'])
    # Define power-of-2 ticks
    nice_widths = [2**i for i in range(0, max_width.bit_length())]  # Covers up to max_width
    log_norm = LogNorm(vmin=min(nice_widths), vmax=max(nice_widths))

    # Create figure and axes
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=log_norm)
    sm.set_array([])

    # Create colorbar
    cbar = fig.colorbar(sm, ax=ax2, label='Width', orientation='vertical', pad=0.01)

    # Set colorbar ticks and labels at powers of 2
    cbar.set_ticks(nice_widths)
    cbar.set_ticklabels([f"$2^{int(np.log2(tick))}$" for tick in nice_widths])

    # Adjust layout and save the figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(cfg.sweep.save_file), exist_ok=True)
    plt.savefig(cfg.sweep.save_file, dpi=300)
    if show:
        plt.show()

def plot_dataset(X, bins, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    ax.hist2d(*X.T, bins=bins)
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set(**kwargs)


@torch.no_grad()
def plot_flow_at_time(flow_model, time, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    points = torch.linspace(-2, 2, 10)
    flow_input = torch.cartesian_prod(points, points)
    flow_output = flow_model(flow_input, time=torch.full(flow_input.shape[:1], time))
    ax.quiver(
        torch.stack(torch.chunk(flow_input[:, 0], len(points))).numpy(),
        torch.stack(torch.chunk(flow_input[:, 1], len(points))).numpy(),
        torch.stack(torch.chunk(flow_output[:, 0], len(points))).numpy(),
        torch.stack(torch.chunk(flow_output[:, 1], len(points))).numpy(),
        scale=len(points),
    )
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax.set(**kwargs)


def animate_flow(flow_model, frames: int = 20, target_file='flow_animation.mp4'):

    def plot_frame(time):
        plt.cla()
        plot_flow_at_time(flow_model, time=time, title=f'flow at time={time:.2f}')
    
    fig = plt.figure(figsize=(8, 8))
    animation = FuncAnimation(fig, plot_frame, frames=np.linspace(0, 1, frames))
    animation.save(target_file, writer='ffmpeg')
    plt.close()
    
@torch.no_grad()
def run_flow(flow_model, x_0, t_0, t_1, device='cpu'):
    def f(t: float, x):
        return flow_model(x, time=torch.full(x.shape[:1], t, device=device))

    return odeint(f, x_0, t_0, t_1, phi=flow_model.parameters())


def plot_data(model, dataset, target_file: str = 'generated_vs_target.png') -> None:
    dataset = next(iter(dataset))
    noise = torch.randn_like(dataset)
    fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    plot_dataset(run_flow(model, noise, 0, 1, device='cpu').cpu(), bins=64, ax=axs[0], title='model generated dataset')
    plot_dataset(dataset, bins=64, ax=axs[1], title='target dataset')
    fig.savefig(target_file)
    plt.close()


def linear_decay_lr(step, num_iterations, learning_rate):
    return learning_rate * (1 - step / num_iterations)


def warmup_cooldown_lr(
    step, num_iterations, learning_rate, warmup_iters, warmdown_iters
):
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    elif step < num_iterations - warmdown_iters:
        return learning_rate
    else:
        decay_ratio = (num_iterations - step) / warmdown_iters
        return learning_rate * decay_ratio
