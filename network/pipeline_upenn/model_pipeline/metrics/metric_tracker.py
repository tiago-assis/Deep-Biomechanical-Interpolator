import numpy as np


class MetricTracker():
    def __init__(self, metric_names, mode='train'):
        assert mode in ['train', 'val', 'test'], f"Metric tracking mode should be in ['train', 'val', 'test']. Got '{mode}'"
        if mode == 'test':
            raise NotImplementedError("Metric tracking not implemented for 'test' mode.")
        self.metrics = metric_names
        self.mode = mode
        self.mode_dict = {'train': 'Training',
                          'val': 'Validation',
                          'test': 'Testing'}
        self.reset_running()
        self.reset_epoch()

    def reset_running(self):
        self.running_totals = {metric: 0.0 for metric in self.metrics}
        self.running_counts = {metric: 0 for metric in self.metrics}
    
    def reset_epoch(self):
        self.epoch_totals = {metric: 0.0 for metric in self.metrics}
        self.epoch_counts = {metric: 0 for metric in self.metrics}
        self.max_error = -np.inf

    def update(self, max_error, **metrics_kwargs):
        self.update_max_error(max_error)
        for metric, value in metrics_kwargs.items():
            if metric in self.metrics:
                self.running_totals[metric] += value
                self.running_counts[metric] += 1
                self.epoch_totals[metric] += value
                self.epoch_counts[metric] += 1
            else:
                raise ValueError(f"Unknown metric: {metric}")

    def update_max_error(self, error):
        if error > self.max_error:
            self.max_error = error
    
    def get_metric_avg(self, metric_type='running'):
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
        epoch = self.get_metric_avg(metric_type="epoch")
        print(f"{self.mode_dict[self.mode]} Epoch Metrics:")
        print(f"Max Absolute Error: {self.max_error}")
        for k,v in epoch.items():
            print(f"{k}: {v:.4f}")
        print("\n")

    def print_running(self):
        running = self.get_metric_avg(metric_type='running')
        print(f"{self.mode_dict[self.mode]} Running Metrics:")
        print(f"Max Absolute Error: {self.max_error}")
        for k,v in running.items():
            print(f"{k}: {v:.4f}")
        print("\n")

    def save_metrics(self, writer, step):
        writer.add_scalar(f"{self.mode}/Max Absolute Error", self.max_error, step)
        running_avg = self.get_metric_avg()
        for metric, value in running_avg.items():
            writer.add_scalar(f"{self.mode}/{metric}", value, step)
        writer.flush()