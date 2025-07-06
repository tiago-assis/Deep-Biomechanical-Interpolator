from typing import List, Dict
import torch
from torch.utils.tensorboard import SummaryWriter


class MetricTracker():
    """
    Tracks training, validation, and testing metrics across steps and epochs.
    Supports average computation, metric printing, and logging to TensorBoard.
    """
    def __init__(self, metric_names: List[str], mode: str = 'train'):
        """
        Args:
            metric_names (List[str]): List of metric names to track.
            mode (str): Indicates current tracking phase. One of ['train', 'val', 'test']. 
        """
        assert mode in ['train', 'val', 'test'], f"Metric tracking mode should be in ['train', 'val', 'test']. Got '{mode}'"
        self.metrics = metric_names
        self.mode = mode
        self.mode_dict = {'train': 'Training',
                          'val': 'Validation',
                          'test': 'Testing'}
        self.reset_running()
        self.reset_epoch()

    def reset_running(self):
        """Resets the running totals and counts for a new step or mini-batch."""
        self.running_totals = {metric: 0.0 for metric in self.metrics}
        self.running_counts = {metric: 0 for metric in self.metrics}
    
    def reset_epoch(self):
        """Resets the epoch totals and counts for a new epoch."""
        self.epoch_totals = {metric: 0.0 for metric in self.metrics}
        self.epoch_counts = {metric: 0 for metric in self.metrics}

    def update(self, **metrics_kwargs: Dict[str, torch.Tensor]):
        """
        Updates metric trackers with new values.

        Args:
            **metrics_kwargs: Keyword arguments mapping metric names to values.
        """
        for metric, value in metrics_kwargs.items():
            if metric in self.metrics:
                self.running_totals[metric] += value.detach().cpu().item()
                self.running_counts[metric] += 1
                self.epoch_totals[metric] += value.detach().cpu().item()
                self.epoch_counts[metric] += 1              
    
    def get_metric_avg(self, metric_type: str = 'running') -> Dict[str, float]:
        """
        Computes average of tracked metrics.

        Args:
            metric_type (str): 'running' for current step, 'epoch' for full epoch.

        Returns:
            avg (Dict[str, float]): Dictionary of averaged metric values.
        """
        assert metric_type in ['running', 'epoch'], f"Metric type to retrieve average should be 'running' or 'epoch'. Got '{metric_type}'"

        if metric_type == 'running':
            avg = {
                metric: self.running_totals[metric] / self.running_counts[metric]
                if self.running_counts[metric] > 0 else 0.0
                for metric in self.metrics
            }
        elif metric_type == 'epoch':
            avg = {
                metric: self.epoch_totals[metric] / self.epoch_counts[metric] 
                if self.epoch_counts[metric] > 0 else 0.0 
                for metric in self.metrics
            }
        return avg
    
    def print_epoch(self):
        """Prints average metrics for the current epoch."""
        epoch = self.get_metric_avg(metric_type="epoch")
        print(f"\n{self.mode_dict[self.mode]} Epoch Metrics:")
        for k,v in epoch.items():
            print(f"{k}: {v:.4f}")
        print("\n")

    def print_running(self):
        """Prints average metrics for the current step."""
        running = self.get_metric_avg(metric_type='running')
        print(f"\n{self.mode_dict[self.mode]} Running Metrics:")
        for k,v in running.items():
            print(f"{k}: {v:.4f}")
        print("\n")

    def save_metrics(self, writer: SummaryWriter, step: int):
        """
        Logs metrics to a TensorBoard writer.

        Args:
            writer (SummaryWriter): TensorBoard SummaryWriter.
            step (int): Global step to log the metrics at.
        """
        running_avg = self.get_metric_avg()
        for metric, value in running_avg.items():
            writer.add_scalar(f"{self.mode}/{metric}", value, step)
        writer.flush()
        