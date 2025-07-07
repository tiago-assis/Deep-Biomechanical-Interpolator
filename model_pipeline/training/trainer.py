from tqdm import tqdm
import math
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple, Dict, Any
from model_pipeline.metrics.metric_tracker import MetricTracker
from model_pipeline.metrics.logger import save_plot
from model_pipeline.metrics.losses import MaskedMSELoss, tr_error, dice_score, hd95, pct_negative_jdet
from model_pipeline.utils.utils import warp_tensor, dilate_binary_mask


def process_batch(batch: Dict[str, torch.Tensor], 
                  model: torch.nn.Module, 
                  mse_w: float = 1.0, 
                  lncc_loss: Optional[Any] = None, 
                  lncc_w: float = 1.0, 
                  reg_penalty: Optional[Any] = None, 
                  reg_w: float = 1.0, 
                  save_warps: bool = False, 
                  extra_eval: bool = False, 
                  save_metrics_every: int = 20, 
                  start_time: float = float('nan'), 
                  device: str = 'cuda') -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]], Optional[int]]:
    """
    Processes a single batch through the model, computes loss components and optional metrics.

    Args:
        batch: Dictionary containing all input tensors.
        model: PyTorch model to generate predictions.
        mse_w: Weight for MSE loss.
        lncc_loss: Optional local cross-correlation loss function.
        lncc_w: Weight for LNCC loss.
        reg_penalty: Optional regularization penalty function.
        reg_w: Weight for regularization penalty.
        save_warps: If True, computes and saves warped images for visualization.
        extra_eval: If True, computes additional evaluation metrics like TRE, Dice, HD95.
        save_metrics_every: Frequency control for saving metrics. Can be a negative number to disable metric computation and saving.
        start_time: Time used for runtime measurement. Timer includes data loading.
        device: Device to use for computation.

    Returns:
        total_loss: Weighted total loss tensor.
        metrics: Optional dictionary of evaluation metrics.
        plot_data: Optional dictionary of tensors for visualization.
        tumor_z_slice: Optional slice index for tumor visualization.
    """
    batch = {k: v.to(device) for k, v in batch.items()}
    time_baseline = time.time() - start_time
    img = batch['img']
    gt_ddf = batch['gt_ddf']
    init_ddf = batch['init_ddf']
    brain_seg = batch['brain_seg']
    tumor_seg = batch['tumor_seg']
    #tumor_seg = dilate_binary_mask(tumor_seg)
    non_tumor_mask = 1.0 - tumor_seg
    upenn_edema_seg = batch['upenn_edema_seg']
    init_ddf = torch.where(brain_seg > 0, init_ddf, 0.0)
    gt_ddf = torch.where(brain_seg > 0, gt_ddf, 0.0)
    
    inputs = torch.cat([init_ddf, img], dim=1)
    pred_ddf = model(inputs)
    time_pred = time.time() - start_time

    masked_mse = MaskedMSELoss()
    mse = masked_mse(pred_ddf, gt_ddf, non_tumor_mask)

    if lncc_loss is not None or save_warps:
        bg = img[0,0,0,0,0]
        img_pre_warped = (torch.where(non_tumor_mask > 0, img, bg)).expand(3, -1, -1, -1, -1).contiguous()
        pre_warped_stack = torch.cat([pred_ddf, gt_ddf, init_ddf], dim=0)
        img_warps = warp_tensor(img_pre_warped, pre_warped_stack, mask=brain_seg, bg_value=bg)

    if lncc_loss is not None:
        lncc = lncc_loss(img_warps[0].unsqueeze(0), img_warps[1].unsqueeze(0))
    else:
        lncc = 0.0

    if reg_penalty:
        brain_wo_tumor = ((brain_seg - tumor_seg) > 0).float()
        reg = reg_penalty(pred_ddf, mask=brain_wo_tumor)
    else:
        reg = 0.0

    total_loss = mse * mse_w + lncc * lncc_w + reg * reg_w
    
    if save_metrics_every > 0:
        with torch.no_grad():
            edema_region_mask = upenn_edema_seg * non_tumor_mask ###
            l2_norm_pred = torch.norm(pred_ddf - gt_ddf, p=2, dim=1) * non_tumor_mask.squeeze(0)
            l2_norm_init = torch.norm(init_ddf - gt_ddf, p=2, dim=1) * non_tumor_mask.squeeze(0)
            metrics = {
                'Total Loss': total_loss,
                'Baseline MSE': masked_mse(init_ddf, gt_ddf, non_tumor_mask),
                'Baseline MSE (edema)': masked_mse(init_ddf, gt_ddf, edema_region_mask),
                'Prediction MSE': mse,
                'Prediction MSE (edema)': masked_mse(pred_ddf, gt_ddf, edema_region_mask), 
                'Baseline Max Error': l2_norm_init.max(),
                'Baseline Max Error (edema)': (l2_norm_init * edema_region_mask).max(),
                'Max Error': l2_norm_pred.max(),
                'Max Error (edema)': (l2_norm_pred * edema_region_mask).max(),       
            }
            if lncc_loss:
                metrics.update({
                    'Baseline LNCC': lncc_loss(img_warps[1].unsqueeze(0), img_warps[2].unsqueeze(0)),
                    'Prediction LNCC': lncc
                })
            if reg_penalty:
                metrics.update({
                    'GT Reg': reg_penalty(gt_ddf, mask=brain_wo_tumor),
                    'GT Reg (edema)': reg_penalty(gt_ddf, mask=edema_region_mask),
                    'GT Reg (tumor)': reg_penalty(gt_ddf, mask=tumor_seg),
                    'Prediction Reg': reg,
                    'Prediction Reg (edema)': reg_penalty(pred_ddf, mask=edema_region_mask),
                    'Prediction Reg (tumor)': reg_penalty(pred_ddf, mask=tumor_seg)
                })
            if extra_eval:
                tre_pred = tr_error(batch['kpts'], gt_ddf, pred_ddf)
                tre_init = tr_error(batch['kpts'], gt_ddf, init_ddf)
                init_warped_brain_seg = warp_tensor(brain_seg, init_ddf)
                pred_warped_brain_seg = warp_tensor(brain_seg, pred_ddf)
                target_warped_brain_seg = warp_tensor(brain_seg, gt_ddf)
                metrics.update({
                    'Baseline TRE': tre_init.mean(),
                    'Prediction TRE': tre_pred.mean(),
                    'Baseline Max TRE': tre_init.max(),
                    'Max TRE': tre_pred.max(),
                    'Baseline Dice': dice_score(init_warped_brain_seg, target_warped_brain_seg).mean(),
                    'Prediction Dice': dice_score(pred_warped_brain_seg, target_warped_brain_seg).mean(),
                    'Baseline HD95': hd95(init_warped_brain_seg, target_warped_brain_seg).mean(),
                    'HD95': hd95(pred_warped_brain_seg, target_warped_brain_seg).mean(),
                    'Baseline %|J|<0': pct_negative_jdet(init_ddf),
                    'Prediction %|J|<0': pct_negative_jdet(pred_ddf),
                    'Baseline Runtime': torch.tensor(time_baseline),
                    'Runtime': torch.tensor(time_pred)
                })
            plot_data = {
                "MRI Scan": img,
                "Pred Disp Field": pred_ddf,
                "GT Disp Field (Sim)": gt_ddf,
                "Init Disp Field (Interp)": init_ddf,
                #"Brain Mask": brain_seg,
                #"Tumor Mask": tumor_seg,
                #"Tumor+Edema Mask": upenn_edema_seg,
                "L2 Distance": l2_norm_pred.unsqueeze(1),
            }
            tumor_z_slice = torch.argwhere(tumor_seg.squeeze(0) > 0).float()
            tumor_z_slice = torch.mean(tumor_z_slice, dim=0)
            tumor_z_slice = int(tumor_z_slice[3].item())
            if save_warps:
                plot_data["Pred Warped MRI"] = img_warps[0].unsqueeze(0)
                plot_data["GT Warped MRI"] = img_warps[1].unsqueeze(0)
                plot_data["Init Warped MRI"] = img_warps[2].unsqueeze(0)
            #if reg_matrix is not None:
                #plot_data['Prediction Reg'] = reg_matrix

        return total_loss, metrics, plot_data, tumor_z_slice
    return total_loss, None, None, None


def run_trainer(dataloader: DataLoader, 
                model: torch.nn.Module, 
                epoch: int, 
                writer: SummaryWriter, 
                metric_tracker: MetricTracker, 
                optimizer: Optional[torch.optim.Optimizer] = None, 
                mode: str = 'train',
                mse_w: float = 1.0, 
                lncc_loss: Optional[Any] = None, 
                lncc_w: float = 1.0, 
                reg_penalty: Optional[Any] = None, 
                reg_w: float = 1.0,
                save_metrics_every: int = 20, 
                local_plot_save: bool = False, 
                save_warps: bool = False, 
                extra_eval: bool = False, 
                device: str = 'cuda') -> Optional[float]:
    is_training = mode.lower() == 'train'
    model.train() if is_training else model.eval()
    metric_tracker.reset_running()
    metric_tracker.reset_epoch()

    dataloader_len = len(dataloader)

    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        start_time = time.time()
        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            total_loss, metrics, plot_data, tumor_z_slice = process_batch(
                batch, model, mse_w, lncc_loss, lncc_w, reg_penalty, reg_w,
                save_warps, extra_eval, save_metrics_every, start_time, device
            )

            if is_training:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            if save_metrics_every > 0:
                metric_tracker.update(**metrics)

            if i % save_metrics_every == save_metrics_every - 1:
                metric_tracker.save_metrics(writer, epoch * dataloader_len + i)
                #metric_tracker.print_running()
                metric_tracker.reset_running()
                save_plot(nrows=math.ceil(len(plot_data) / 4),
                          ncols=4,
                          mode=mode,
                          writer=writer,
                          step=epoch * dataloader_len + i,
                          local_save=local_plot_save,
                          slice_num=tumor_z_slice,
                          **plot_data)
            
            start_time = time.time()

    if save_metrics_every > 0:
        metric_tracker.print_epoch()

    avg_epoch_mse = metric_tracker.get_metric_avg(metric_type='epoch')['Prediction MSE']

    return avg_epoch_mse if mode == 'val' else None
