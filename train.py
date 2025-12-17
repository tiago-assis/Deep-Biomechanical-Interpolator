import argparse
import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from monai.utils.misc import set_determinism
#from model_pipeline.networks.vxm.model import VoxelMorph
from model_pipeline.networks.unet3d.model import ResidualUNetSE3D, ResidualUNet3D, UNet3D
from model_pipeline.datatools.loader import generate_datasets, get_dataloaders
from model_pipeline.metrics.logger import save_configs, get_writer, get_metric_tracker
from model_pipeline.metrics.losses import LNCC, JacobianDetLoss, BendingEnergyLoss
from model_pipeline.training.trainer import run_trainer
from model_pipeline.utils.utils import save_checkpoint, load_checkpoint


IN_CHANNELS = 4

def set_configs() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for configuring the training of a 3D UNet model.

    Returns:
        argparse.Namespace: Parsed command-line arguments as attributes.
    """
    parser = argparse.ArgumentParser(
        description="Train a model for biomechanically accurate dense displacement field interpolation using preoperative MRI scans and sparse displacements at matched keypoints, supervised with biomechanical simulation as synthetic ground truths."
    )
    parser.add_argument('--data', type=str, default=None, help='Path to the data directory.')
    parser.add_argument('--test', action='store_true', help='Whether to run the model on the test set only.')
    parser.add_argument('--checkpoint', action='store_true', help='Save additional checkpoints during training. By default, checkpoints are automatically saved when the validation loss improves. This increases the frequency of checkpointing to every [--evaluate_every] epochs.')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load a saved model checkpoint.')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save training checkpoints.')
    parser.add_argument('--configs', type=str, default=None, help='JSON file to override CLI arguments.')
    parser.add_argument('--split', type=float, default=[0.10, 0.15], help='Validation and test data split ratios. The first value is the ratio of cases in the validation set, and the second value is the ratio of cases in the test set.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (currently only 1 supported).')
    parser.add_argument('--size', type=int, nargs=3, default=(160,192,144), help='Input tensor size to the model.')
    parser.add_argument('--model', type=str, default='res-unet-se', choices=['vxm', 'unet', 'res-unet', 'res-unet-se'], help='Model architecture to use (vxm, unet, res-unet, res-unet-se).')
    parser.add_argument('--predict_residual', action='store_true', help='Predict residuals `x + f(x,y)` or full displacements `f(x,y)`.')
    parser.add_argument('--layer_order', type=str, default='cil', help='Order of components in convolution blocks. Check ./model_pipeline/networks/unet3d/buildingblocks.py for options.')
    parser.add_argument('--num_features', '--nfeats', default="32", help='Feature channels per level. If only an integer is provided, the features will double at each level. Alternatively, provide a comma-separated string of integers to specify the exact number of features at each level (e.g., "16,32,32,32").')
    parser.add_argument('--num_levels', '--nlevels', type=int, default=4, help='Encoder/decoder levels.')
    parser.add_argument('--num_groups', '--ngroups', type=int, default=8, help='Groups for GroupNorm.')
    parser.add_argument('--se_mode', type=str, default='scse', choices=['scse', 'cse', 'sse'], help='Squeeze-and-Excitation module (scse, cse, or sse).')
    parser.add_argument('--upsample_mode', type=str, default='nearest', choices=['nearest', 'trilinear'], help='Upsampling method for the decoder (nearest or trilinear).')
    parser.add_argument('--pool_mode', type=str, default='max', choices=['max', 'avg'], help='Pooling method for the encoder (max or avg).')
    parser.add_argument('--mse_w', type=float, default=1.0, help='Weight for the MSE loss.')
    parser.add_argument('-e', '--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--lncc', action='store_true', help='Flag to use the LNCC loss.')
    parser.add_argument('--lncc_w', type=float, default=1.0, help='Weight for the LNCC loss.')
    parser.add_argument('--lncc_window', type=int, default=2, help='LNCC window size.')
    parser.add_argument('--be', '--hess', '--hessian', action='store_true', help='Flag to use bending energy penalty.')
    parser.add_argument('--jdet', action='store_true', help='Flag to use Jacobian determinant regularization.')
    parser.add_argument('--jdet_mode', type=str, default='negative', choices=['negative', 'unit'], help='Jacobian mode (`negative` or `unit`).')
    parser.add_argument('--reg_w', type=float, default=1.0, help='Regularization weight.')
    parser.add_argument('--extra_eval', action='store_true', help='Flag to additionally compute TRE, Dice, HD95 metrics during validation.')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation for the preoperative scans (intensity only).')
    parser.add_argument('--evaluate_every', type=int, default=1, help='Evaluate model every N epochs.')
    parser.add_argument('--train_save_every', type=int, default=121, help='Save training metrics every N batches. If N > total number of cases, then the training metrics are saved every epoch. If N < 0, saving training metrics is disabled.')
    parser.add_argument('--local_plot_save', action='store_true', help='Flag to save training plots locally. By default, plots are only saved to TensorBoard.')
    parser.add_argument('--save_warps', action='store_true', help='Flag to save warped image outputs to TensorBoard for visualization.')
    parser.add_argument('--metrics_dir', type=str, default='metric_saves', help='Directory to save metrics/plots.')
    parser.add_argument('--interp', type=str, default=None, choices=['tps', 'linear'], help='Initial interpolation method (`tps`, `linear`, None). If None, the interpolation method is randomly chosen.')
    parser.add_argument('--num_kpts', type=int, default=None, help='Number of keypoints to sample from a list of keypoint coordinates and corresponding displacements. Overrides the minimum and maximum number of keypoints flags.')
    parser.add_argument('--min_kpts', type=int, default=20, help='Minimum number of keypoints to sample from a list of keypoint coordinates and corresponding displacements.')
    parser.add_argument('--max_kpts', type=int, default=20, help='Maximum number of keypoints to sample from a list of keypoint coordinates and corresponding displacements.')
    parser.add_argument('--run_prefix', type=str, default='', help='Filename prefix for saved run files (TensorBoard).')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model (cuda or cpu).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')

    args = parser.parse_args()

    if args.batch_size > 1:
        raise NotImplementedError("Batch sizes greater than 1 are not yet supported.")
    
    if args.configs is not None:
        with open(args.configs, 'r') as f:
            config_args = json.load(f)
        for key, value in config_args.items():
            setattr(args, key, value)

    if isinstance(args.num_features, str):
        if "," in args.num_features:
            args.num_features = [int(f) for f in args.num_features.split(",")]
            args.num_levels = len(args.num_features)
        else:
            args.num_features = int(args.num_features)

    if args.num_kpts is not None:
        args.min_kpts = args.num_kpts
        args.max_kpts = args.num_kpts

    assert args.data is not None, "Data path must be provided."
    assert not (args.test and args.load_model is None), "If testing, a model checkpoint must be provided."
    assert os.path.exists(args.data) and os.path.isdir(args.data), f"Data path provided is not a valid directory: {args.data}"
    assert args.epochs > 0, "Number of epochs must be a positive integer."
    assert 2 <= len(args.split) <= 3 and 0 < sum(args.split) < 1.0, f"Data split ratios need to be in the format [(train_split, )val_split, test_split] and its sum must be in the interval (0, 1)."
    assert args.lr > 0, "Learning rate must be a positive float."
    assert args.lncc_window > 0, "LNCC window size must be a positive integer."
    assert args.evaluate_every > 0, "Evaluation frequency must be a positive integer."
    #assert args.train_save_every > 0, "Training metrics save frequency must be a positive integer."
    #assert args.val_save_every > 0, "Validation metrics save frequency must be a positive integer."
    assert args.size[0] > 0 and args.size[1] > 0 and args.size[2] > 0, "Input size dimensions must be positive integers."
    assert args.interp is None or args.interp in ['tps', 'linear'], f"Interpolation method must be either 'tps', 'linear', or None; got '{args.interp}'"
    assert args.min_kpts >= 5, "Minimum number of sampled keypoints must be >= 5."
    assert args.max_kpts >= args.min_kpts, "The number of maximum sampled keypoints must be an integer >= the number of minimum sampled keypoints."
    assert (args.seed is None) or args.seed >= 0, f"Deterministic seed needs to be a non-negative integer or None; got {args.seed}"

    return args


if __name__ == "__main__":
    args = set_configs()

    writer, log_dir = get_writer(
        path=os.path.join(args.metrics_dir, 'runs'),
        run_prefix=args.run_prefix
    )
    save_configs(log_dir, vars(args))

    set_determinism(args.seed)

    train_dataset, val_dataset, test_dataset = generate_datasets(
        data_path=args.data,
        split=args.split,
        augment=args.augment,
        interp=args.interp,
        min_kpts=args.min_kpts,
        max_kpts=args.max_kpts,
        size=args.size,
        device=args.device,
        kpts_sampling_seed=args.seed,
        ddf_sampling_seed=args.seed,
        data_split_seed=args.seed
    )
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,  # Currently only supports batch size of 1
        shuffle=True,
        dataloader_generator=torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    )

    if args.train_save_every > len(train_dataloader):
        args.train_save_every = len(train_dataloader)

    metrics = [
        'Total Loss',
        'Baseline MSE',
        'Baseline MSE (edema)',
        'Prediction MSE',
        'Prediction MSE (edema)',
        'Baseline Max Error',
        'Baseline Max Error (edema)',
        'Max Error',
        'Max Error (edema)'
    ]
    if args.lncc:
        metrics += ['Baseline LNCC', 'Prediction LNCC']
    if args.jdet or args.be:
        metrics += [
            'GT Reg',
            'GT Reg (edema)',
            'GT Reg (tumor)',
            'Prediction Reg',
            'Prediction Reg (edema)',
            'Prediction Reg (tumor)'
        ]
    if args.extra_eval:
        metrics_val = metrics.copy()
        metrics_val += [
            'Baseline TRE',
            'Prediction TRE',
            'Baseline Max TRE',
            'Max TRE',
            'Baseline Dice',
            'Prediction Dice',
            'Baseline HD95',
            'HD95',
            'Baseline %|J|<0',
            'Prediction %|J|<0',
            'Baseline Runtime',
            'Runtime'
        ]
    
    train_metric_tracker = get_metric_tracker(
        metrics=metrics,
        mode='train'
    )

    val_metric_tracker = get_metric_tracker(
        metrics=metrics_val if args.extra_eval else metrics,
        mode='val'
    )

    test_metric_tracker = get_metric_tracker(
        metrics=metrics_val if args.extra_eval else metrics,
        mode='test'
    )

    if args.model == 'vxm':
        raise NotImplementedError
        #model = VoxelMorph(infeats=IN_CHANNELS, nb_features=args.num_features, predict_residual=args.predict_residual)
    elif args.model == 'unet':
        model = UNet3D(
            in_channels=IN_CHANNELS,
            out_channels=3,
            final_sigmoid=False,
            f_maps=args.num_features,
            layer_order=args.layer_order,
            num_groups=args.num_groups,
            num_levels=args.num_levels,
            is_segmentation=False, 
            predict_residual=args.predict_residual,
            pool_type=args.pool_mode,
            upsample_mode=args.upsample_mode
        )
    elif args.model == 'res-unet':
        model = ResidualUNet3D(
            in_channels=IN_CHANNELS,
            out_channels=3,
            final_sigmoid=False,
            f_maps=args.num_features,
            layer_order=args.layer_order,
            num_groups=args.num_groups,
            num_levels=args.num_levels,
            is_segmentation=False, 
            predict_residual=args.predict_residual,
            pool_type=args.pool_mode
        )
    else:
        model = ResidualUNetSE3D(
            in_channels=IN_CHANNELS,
            out_channels=3,
            final_sigmoid=False,
            f_maps=args.num_features,
            layer_order=args.layer_order,
            num_groups=args.num_groups,
            num_levels=args.num_levels,
            is_segmentation=False, 
            predict_residual=args.predict_residual,
            pool_type=args.pool_mode,
            se_module=args.se_mode
        )

    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr/100)
    #scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs, power=0.4)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=0.001, threshold_mode='abs', cooldown=0, min_lr=1e-6)

    if args.lncc:
        lncc_loss = LNCC(args.lncc_window).to(args.device)
    else:
        lncc_loss = None

    if args.be:
        reg_penalty = BendingEnergyLoss()
    elif args.jdet:
        reg_penalty = JacobianDetLoss(mode=args.jdet_mode)
    else:
        reg_penalty = None

    init_epoch = 0
    best_loss = torch.inf
    if args.load_model is not None:
        checkpoint = load_checkpoint(args.load_model, model, optimizer, verbose=True)
        init_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']

    if not args.test:
        print(f"\nA {args.model} model was selected to train with the {'GPU' if args.device == 'cuda' else 'CPU'}.")
        print(f"Metrics and setup arguments are being saved to: {log_dir}")
        print(f"Model checkpoints are being saved to: {args.save_dir}")
        print(f"\nTraining model...\n")
        for epoch in tqdm(range(init_epoch, args.epochs)):
            run_trainer(
                train_dataloader, 
                model, 
                epoch, 
                writer, 
                train_metric_tracker, 
                optimizer=optimizer, 
                mode='train',
                mse_w=args.mse_w, 
                lncc_loss=lncc_loss, 
                lncc_w=args.lncc_w, 
                reg_penalty=reg_penalty, 
                reg_w=args.reg_w,
                save_metrics_every=args.train_save_every, 
                local_plot_save=args.local_plot_save, 
                save_warps=args.save_warps,
                device=args.device
            )

            if epoch % args.evaluate_every == args.evaluate_every - 1:
                print(f"\n\nValidating model...")
                loss = run_trainer(
                    val_dataloader, 
                    model, 
                    epoch, 
                    writer, 
                    val_metric_tracker, 
                    mode='val',
                    mse_w=args.mse_w, 
                    lncc_loss=lncc_loss, 
                    lncc_w=args.lncc_w, 
                    reg_penalty=reg_penalty, 
                    reg_w=args.reg_w,
                    extra_eval=args.extra_eval,
                    save_metrics_every=len(val_dataloader), 
                    local_plot_save=args.local_plot_save, 
                    save_warps=args.save_warps, 
                    device=args.device
                )

                is_best = loss < best_loss
                best_loss = loss if is_best else best_loss
                if is_best or args.checkpoint:
                    save_checkpoint(model, optimizer, epoch, loss, is_best=is_best, filename=args.run_prefix, save_dir=args.save_dir, additional_info=None)

            scheduler.step()

        if args.epochs % args.evaluate_every != 0:
            loss = run_trainer(
                val_dataloader, 
                model, 
                epoch, 
                writer, 
                val_metric_tracker, 
                mode='val',
                mse_w=args.mse_w, 
                lncc_loss=lncc_loss, 
                lncc_w=args.lncc_w, 
                reg_penalty=reg_penalty, 
                reg_w=args.reg_w,
                extra_eval=args.extra_eval,
                save_metrics_every=len(val_dataloader), 
                local_plot_save=args.local_plot_save, 
                save_warps=args.save_warps, 
                device=args.device
            )
            
            is_best = loss < best_loss
            best_loss = loss if is_best else best_loss
            if is_best or args.checkpoint:
                save_checkpoint(model, optimizer, epoch, loss, is_best=is_best, filename=args.run_prefix, save_dir=args.save_dir, additional_info=None)

        best_model = glob.glob(os.path.join(args.save_dir, "best_checkpoints", f"{args.run_prefix}*"))[0]
        checkpoint = load_checkpoint(best_model, model, optimizer, verbose=True)
        init_epoch = checkpoint['epoch']
        print("\nRunning the best model on the test set...\n")
    
    run_trainer(
        test_dataloader, 
        model, 
        init_epoch, 
        writer, 
        test_metric_tracker,
        mode='test',
        mse_w=args.mse_w, 
        lncc_loss=lncc_loss, 
        lncc_w=args.lncc_w, 
        reg_penalty=reg_penalty, 
        reg_w=args.reg_w,
        save_metrics_every=1, 
        local_plot_save=args.local_plot_save, 
        save_warps=args.save_warps,
        extra_eval=args.extra_eval,
        device=args.device
    )

    writer.flush()
    writer.close()
    
