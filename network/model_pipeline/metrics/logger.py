import os
import json
import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model_pipeline.metrics.metric_tracker import MetricTracker


def save_configs(log_dir, args):
    with open(os.path.join(log_dir, "setup_arguments.conf"), "w") as f:
        json.dump(args, f, indent=4)

def get_writer(path, run_prefix="run"):
    dt = datetime.datetime.now()
    run_name = run_prefix + "_" if not run_prefix.endswith("_") and run_prefix != "" else run_prefix
    run_name += f"{dt.strftime('%Y-%m-%d')}_{dt.strftime('%H-%M-%S')}"
    writer = SummaryWriter(os.path.join(path, run_name))
    return writer, writer.log_dir

def get_metric_tracker(metrics, mode):
    assert isinstance(metrics, list) and len(metrics) > 0, "Metrics must be a non-empty list of custom metric names."
    assert mode in ['train', 'val'], f"Mode must be either 'train','val', or 'test'; got {mode}"
    if mode == 'test':
        raise NotImplementedError("Test mode not implemented yet.")
    
    metric_tracker = MetricTracker(metrics, mode=mode)
    return metric_tracker

def save_plot(nrows, ncols, writer, step, mode='val', slice_num=80, figsize=(20,10), local_save=False, return_plot=False, **plot_data):
    assert mode in ['train', 'val'], f"Mode must be either 'Train' or 'Val'; got {mode}"

    fig = plt.figure(figsize=figsize)

    for i, (name, array) in enumerate(plot_data.items()):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.set_title(name)
        cmap = 'gray' if ('MRI' in name or 'Mask' in name or 'Warped' in name) else 'RdBu'

        array = array.detach().cpu().squeeze(0).numpy().transpose(0,3,2,1)

        plt.imshow(array[0,slice_num,:,:], cmap=cmap)
        if 'MRI' not in name and 'Mask' not in name and 'Warped' not in name:
            plt.colorbar()

    writer.add_figure(f'{mode}/Plots', fig, step)
    writer.flush()

    if local_save:
        fig.savefig(os.path.join(writer.log_dir, f'plot_step_{step}'))

    if not return_plot:
        plt.close(fig)
    else:
        return fig
    