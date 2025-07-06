import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import Namespace
from typing import Union, Tuple, List, Dict
from torch.utils.tensorboard import SummaryWriter
from model_pipeline.metrics.metric_tracker import MetricTracker


def save_configs(log_dir: str, args: Union[Namespace, dict]):
    """
    Saves the configuration arguments to a JSON file.

    Args:
        log_dir (str): Directory where the configuration file will be saved.
        args (Union[Namespace, dict]): The configuration arguments from argparse.
    """
    with open(os.path.join(log_dir, "setup_arguments.cfg"), "w") as f:
        json.dump(args, f, indent=4)

def get_writer(path: str, run_prefix: str = "run") -> Tuple[SummaryWriter, str]:
    """
    Creates a TensorBoard SummaryWriter with a timestamped run directory.

    Args:
        path (str): Base path to create the run directory.
        run_prefix (str, optional): Prefix for the run name. Defaults to "run".

    Returns:
        writer, log_dir (Tuple[SummaryWriter, str]): The SummaryWriter object and its log directory.
    """
    dt = datetime.datetime.now()
    run_name = run_prefix + "_" if not run_prefix.endswith("_") and run_prefix != "" else run_prefix
    run_name += f"{dt.strftime('%Y-%m-%d')}_{dt.strftime('%H-%M-%S')}"
    writer = SummaryWriter(os.path.join(path, run_name))
    log_dir = writer.log_dir
    return writer, log_dir

def get_metric_tracker(metrics: List[str], mode: str) -> MetricTracker:
    """
    Initializes a MetricTracker for the specified mode and metrics.

    Args:
        metrics (List[str]): List of metric names to track.
        mode (str): One of 'train', 'val', or 'test'.

    Returns:
        metric_tracker (MetricTracker): An instance of the metric tracker configured for the specified mode.
    """
    assert isinstance(metrics, list) and len(metrics) > 0, "Metrics must be a non-empty list of custom metric names."
    assert mode in ['train', 'val', 'test'], f"Mode must be either 'train','val', or 'test'; got {mode}"
    
    metric_tracker = MetricTracker(metrics, mode=mode)
    return metric_tracker

def save_plot(nrows: int, 
              ncols: int, 
              writer: SummaryWriter, 
              step: int, 
              mode: str = 'val', 
              slice_num: int = 80, 
              figsize: Tuple[int, int] = (20,10), 
              local_save: bool = False, 
              **plot_data: Dict[str, torch.Tensor]):
    """
    Plots and logs visual data to TensorBoard.

    Args:
        nrows (int): Number of rows in subplot grid.
        ncols (int): Number of columns in subplot grid.
        writer (SummaryWriter): TensorBoard writer.
        step (int): Step number for logging.
        mode (str): One of 'train', 'val', or 'test'.
        slice_num (int): Slice index to display.
        figsize (Tuple[int, int]): Size of the entire figure.
        local_save (bool): Whether to save the figure locally.
        **plot_data (Dict[str, torch.Tensor]): Named tensors to plot.
    """
    assert mode in ['train', 'val', 'test'], f"Mode must be either 'train','val', or 'test'; got {mode}"

    fig = plt.figure(figsize=figsize)
    for i, (name, array) in enumerate(plot_data.items()):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.set_title(name)
        array = array.detach().squeeze(0).cpu().numpy().transpose(0,3,2,1)
        if ('MRI' in name or 'Mask' in name or 'Warped' in name):
            cmap = 'gray'
            max_data = np.percentile(array, 99.95)
            array[array > max_data] = max_data
            plt.imshow(array[0,slice_num,:,:], cmap=cmap)
        elif "Disp" in name:
            cmap = 'magma'
            dx = array[0, slice_num]
            dy = array[1, slice_num]
            mag = np.sqrt(dx**2 + dy**2)
            plt.imshow(mag, cmap=cmap)
            plt.colorbar()
        else:
            cmap = 'magma'
            plt.imshow(array[0,slice_num,:,:], cmap=cmap)
            plt.colorbar()


    writer.add_figure(f'{mode}/Plots', fig, step)
    writer.flush()

    if local_save:
        fig.savefig(os.path.join(writer.log_dir, f'plot_step_{step}'))

    plt.close(fig)
    