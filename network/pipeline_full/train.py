import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model_pipeline.network.model import UNet
from model_pipeline.datatools.loader import generate_datasets, generate_dataloaders
from model_pipeline.metrics.logger import save_configs, get_writer, get_metric_tracker
from model_pipeline.metrics.losses import LNCC
from model_pipeline.training.trainer import train, evaluate

IN_CHANNELS = 4

def set_configs():
    parser = argparse.ArgumentParser(description="Train a 3D UNet model for dense displacement field prediction from MRI scans and sparse displacements.")
    parser.add_argument('--data', type=str, default=None, help='Path to the data directory.')
    parser.add_argument('--configs', type=str, default=None, help='Path to a JSON file containing the setup arguments. Used to load or override the command line arguments.')
    parser.add_argument('--split', type=float, nargs=2, default=[0.85, 0.15], help='Data split ratios for training and validation sets.')
    #parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training.')  # Currently only supports batch size of 1 due to LinearInterp
    parser.add_argument('--size', type=int, nargs=3, default=(192, 224, 160), help='Input tensor sizes (D, H, W).')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--mse_w', type=float, default=1.0, help='Weight for the MSE loss term.')
    parser.add_argument('--lncc', action='store_true', help='Whether to use LNCC loss as an additional metric.')
    parser.add_argument('--lncc_w', type=float, default=1.0, help='Weight for the LNCC loss term.')
    parser.add_argument('--lncc_window', type=int, default=2, help='Window size for the LNCC loss calculation.')
    parser.add_argument('--hessian', action='store_true', help='Whether to use the Hessian penalty as a regularization term.')
    parser.add_argument('--jdet', action='store_true', help='Whether to use the Jacobian determinant as a regularization term.')
    parser.add_argument('--reg_w', type=float, default=0.1, help='Weight for the regularization term.')
    parser.add_argument('--evaluate_every', type=int, default=3, help='Frequency of validation during training. Sets the number of epochs after which the model is evaluated on the validation set.')
    parser.add_argument('--train_save_every', type=int, default=20, help='Frequency of saving metrics during training. Sets the number of batches after which the metrics are saved.')
    parser.add_argument('--val_save_every', type=int, default=4, help='Frequency of saving metrics during validation. Sets the number of batches after which the metrics are saved.')
    parser.add_argument('--local_plot_save', action='store_true', help='Whether to save training plots locally.')
    parser.add_argument('--metrics_out_path', type=str, default='saves', help='Path to save the metrics and logs.')
    parser.add_argument('--interp', type=str, default='tps', choices=['tps', 'linear'], help='Interpolation mode for the initial displacement field.')
    parser.add_argument('--min_kpts', type=int, default=15, help='Minimum number of sampled keypoints for the displacement field.')
    parser.add_argument('--max_kpts', type=int, default=20, help='Maximum number of sampled keypoints for the displacement field.')
    parser.add_argument('--run_prefix', type=str, default='', help='Information to be prepended to the run name when saving metrics and configs.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')

    args = parser.parse_args()

    if args.configs is not None:
        with open(args.configs, 'r') as f:
            config_args = json.load(f)
        for key, value in config_args.items():
            setattr(args, key, value)

    assert args.data is not None, "Data path must be provided."
    assert os.path.exists(args.data) and os.path.isdir(args.data), f"Data path provided is not a valid directory: '{args.data}'"
    assert sum(args.split) == 1.0, f"Data split ratio must sum to 1; got {args.split}"
    assert isinstance(args.run_prefix, str), "Run prefix must be a string."
    assert isinstance(args.epochs, int) and args.epochs > 0, "Number of epochs must be a positive integer."
    assert isinstance(args.lr, float) and args.lr > 0, "Learning rate must be a positive float."
    assert isinstance(args.lncc_window, int) and args.lncc_window > 0, "LNCC window size must be a positive integer."
    assert args.evaluate_every > 0, "Evaluation frequency must be a positive integer."
    assert args.train_save_every > 0, "Training metrics save frequency must be a positive integer."
    assert args.val_save_every > 0, "Validation metrics save frequency must be a positive integer."
    assert args.size[0] > 0 and args.size[1] > 0 and args.size[2] > 0, "Input size dimensions must be positive integers."
    assert args.interp in ['tps', 'linear'], f"Interpolation method must be either 'tps' or 'linear'; got '{args.interp}'"
    assert isinstance(args.min_kpts, int) and args.min_kpts >= 5, "Minimum sampled keypoints must be an integer >= 5."
    assert isinstance(args.max_kpts, int) and args.max_kpts >= args.min_kpts, "Maximum sampled keypoints must be an integer >= minimum sampled keypoints."
    assert (isinstance(args.seed, int) and args.seed >= 0) or (args.seed is None), f"Deterministic seed needs to be a non-negative integer or None; got {args.seed}"

    os.makedirs(args.metrics_out_path, exist_ok=True)

    return args


if __name__ == "__main__":
    args = set_configs()

    writer, log_dir = get_writer(
        path=os.path.join(args.metrics_out_path, 'runs'),
        run_prefix=args.run_prefix
    )
    save_configs(log_dir, vars(args))

    train_dataset, val_dataset = generate_datasets(
        data_path=args.data,
        data_split=args.split,
        interp=args.interp,
        min_kpts=args.min_kpts,
        max_kpts=args.max_kpts,
        size=args.size,
        device=args.device,
        kpts_sampling_seed=None,
        data_split_generator=torch.Generator().manual_seed(13) #torch.Generator().manual_seed(seed) if seed is not None else None
    )
    train_dataloader, val_dataloader = generate_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=1,  # Currently only supports batch size of 1 due to LinearInterp
        shuffle=True
    )

    if args.lncc:
        metrics = [
            'Initial MSE',
            'Initial LNCC',
            'Prediction MSE',
            'Prediction LNCC',
            'Regularizer',
            'Total Loss'
        ]
    else:
        metrics = [
            'Initial MSE',
            'Prediction MSE',
            'Regularizer',
            'Total Loss'
        ]
    train_metric_tracker = get_metric_tracker(
        metrics=metrics,
        mode='train'
    )
    val_metric_tracker = get_metric_tracker(
        metrics=metrics,
        mode='val'
    )

    model = UNet(args.size, IN_CHANNELS).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr/50) # 1e-6
    mse_loss = nn.MSELoss()
    if args.lncc:
        lncc_loss = LNCC(args.lncc_window).to(args.device)
    else:
        lncc_loss = None


    for epoch in tqdm(range(args.epochs)):    
        train(
            train_dataloader, 
            model, 
            mse_loss, 
            optimizer, 
            epoch, 
            writer, 
            train_metric_tracker,
            mse_w=args.mse_w, 
            lncc_loss=lncc_loss, 
            lncc_w=args.lncc_w,
            hessian=args.hessian, 
            jdet=args.jdet, 
            reg_w=args.reg_w, 
            save_metrics_every=args.train_save_every,
            device=args.device
        )
            
        if epoch % args.evaluate_every == args.evaluate_every - 1:
            evaluate(
                val_dataloader,
                model, 
                mse_loss,
                epoch, 
                writer, 
                val_metric_tracker,
                mse_w=args.mse_w, 
                lncc_loss=lncc_loss, 
                lncc_w=args.lncc_w,
                hessian=args.hessian, 
                jdet=args.jdet, 
                reg_w=args.reg_w, 
                save_metrics_every=args.val_save_every, 
                local_plot_save=args.local_plot_save, 
                device=args.device
            )
        #scheduler.step()

    if args.epochs % args.evaluate_every != 0:
        evaluate(
            val_dataloader,
            model, 
            mse_loss,
            epoch,
            writer, 
            val_metric_tracker,
            mse_w=args.mse_w, 
            lncc_loss=lncc_loss, 
            lncc_w=args.lncc_w,
            hessian=args.hessian, 
            jdet=args.jdet, 
            reg_w=args.reg_w, 
            save_metrics_every=args.val_save_every, 
            local_plot_save=args.local_plot_save, 
            device=args.device
        )

    writer.flush()
    writer.close()
