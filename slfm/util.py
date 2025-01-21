import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from zuko.utils import odeint

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
