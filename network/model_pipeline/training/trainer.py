from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_pipeline.metrics.logger import save_plot
from model_pipeline.metrics.losses import MaskedMSELoss
from model_pipeline.utils.utils import warp_tensor, dilate_binary_mask


def process_batch(batch, model, mse_w=1.0, lncc_loss=None, lncc_w=1.0, reg_penalty=None, reg_w=1.0, save_warps=False, device='cuda'):
    batch = {k: v.to(device) for k, v in batch.items()}
    img = batch['img']
    gt_ddf = batch['gt_ddf']
    init_ddf = batch['init_ddf']
    brain_seg = batch['brain_seg']
    tumor_seg = batch['tumor_seg']
    tumor_seg = dilate_binary_mask(tumor_seg)
    non_tumor_mask = 1.0 - tumor_seg
    upenn_edema_seg = batch['upenn_edema_seg']
    kpt_heatmap = batch.get('kpt_heatmap')
    init_ddf = torch.where(brain_seg > 0, init_ddf, 0.0)
    gt_ddf = torch.where(brain_seg > 0, gt_ddf, 0.0)
    
    if kpt_heatmap is not None:
        kpt_heatmap = torch.where(brain_seg > 0, kpt_heatmap, 0.0)
        inputs = torch.cat([init_ddf, kpt_heatmap, img], dim=1)
    else:
        inputs = torch.cat([init_ddf, img], dim=1)

    pred_ddf = model(inputs)
    #pred_ddf = model(init_ddf)
    
    #mse_loss = nn.MSELoss()
    masked_mse = MaskedMSELoss()
    mse = masked_mse(pred_ddf, gt_ddf, non_tumor_mask) #* mse_w

    bg = img[0,0,0,0,0]
    if lncc_loss is not None or save_warps:
        #img_pre_warped = img * non_tumor_mask ####
        img_pre_warped = (torch.where(non_tumor_mask > 0, img, bg)).expand(3, -1, -1, -1, -1).contiguous()
        #pred_pre_warped = pred_ddf * brain_seg
        #gt_pre_warped = gt_ddf * brain_seg
        #init_pre_warped = init_ddf * brain_seg
        pre_warped_stack = torch.cat([pred_ddf, gt_ddf, init_ddf], dim=0)
        
        img_warps = warp_tensor(img_pre_warped, pre_warped_stack, mask=brain_seg, bg_value=bg)

    if lncc_loss is not None:
        lncc = lncc_loss(img_warps[0].unsqueeze(0), img_warps[1].unsqueeze(0))# * lncc_w 
    else:
        lncc = torch.tensor(0.0, device=device)

    if reg_penalty:
        brain_wo_tumor = ((brain_seg - tumor_seg) > 0).float()
        reg, reg_matrix = reg_penalty(pred_ddf, mask=brain_wo_tumor, return_matrix=True)
        #reg *= reg_w
    else:
        reg = torch.tensor(0.0, device=device)
        reg_matrix = None

    total_loss = mse * mse_w + lncc * lncc_w + reg * reg_w
    
    with torch.no_grad():
        l2_norm = torch.norm(pred_ddf - gt_ddf, p=2, dim=1) * non_tumor_mask.squeeze(0)
        max_error = l2_norm.max()
        metrics = {
            'Total Loss': total_loss,
            'Initial MSE': masked_mse(init_ddf, gt_ddf, non_tumor_mask),# * mse_w,
            'Initial MSE (edema+tumor)': masked_mse(init_ddf, gt_ddf, upenn_edema_seg),# * mse_w,
            'Prediction MSE': mse,
            'Prediction MSE (edema+tumor)': masked_mse(pred_ddf, gt_ddf, upenn_edema_seg)# * mse_w,
        }
        if lncc_loss:
            metrics.update({
                'Initial LNCC': lncc_loss(img_warps[1].unsqueeze(0), img_warps[2].unsqueeze(0)),# * lncc_w,
                'Prediction LNCC': lncc
            })
        if reg_penalty:
            metrics.update({
                'GT Reg': reg_penalty(gt_ddf, mask=brain_wo_tumor),# * reg_w,
                'GT Reg (edema+tumor)': reg_penalty(gt_ddf, mask=upenn_edema_seg),# * reg_w,
                'GT Reg (tumor)': reg_penalty(gt_ddf, mask=tumor_seg),# * reg_w,
                'Regularizer': reg,
                'Regularizer (edema+tumor)': reg_penalty(pred_ddf, mask=upenn_edema_seg),# * reg_w,
                'Regularizer (tumor)': reg_penalty(pred_ddf, mask=tumor_seg)# * reg_w,
            })

        plot_data = {
            "MRI Scan": img,
            "Pred Disp Field": pred_ddf,
            "GT Disp Field (Sim)": gt_ddf,
            "Init Disp Field (Interp)": init_ddf,
            #"Brain Mask": brain_seg,
            #"Tumor Mask": tumor_seg,
            #"Tumor+Edema Mask": upenn_edema_seg,
            "L2 Distance": l2_norm.unsqueeze(1),
        }
        if save_warps:
            plot_data["Pred Warped MRI"] = img_warps[0].unsqueeze(0)
            plot_data["GT Warped MRI"] = img_warps[1].unsqueeze(0)
            plot_data["Init Warped MRI"] = img_warps[2].unsqueeze(0)
        if reg_matrix is not None:
            plot_data['Regularizer'] = reg_matrix #torch.where(torch.where(reg_matrix < 0.2, reg_matrix, 0.0) > 0, reg_matrix, 0.0)
        if kpt_heatmap is not None:
            plot_data['Keypoint Heatmap'] = kpt_heatmap

    return total_loss, max_error, metrics, plot_data


def run_trainer(dataloader, model, epoch, writer, metric_tracker, optimizer=None, mode='train',
              mse_w=1.0, lncc_loss=None, lncc_w=1.0, reg_penalty=None, reg_w=1.0,
              save_metrics_every=20, local_plot_save=False, save_warps=False, device='cuda'):
    is_training = mode.lower() == 'train'
    model.train() if is_training else model.eval()
    metric_tracker.reset_running()
    metric_tracker.reset_epoch()

    dataloader_len = len(dataloader)

    context = torch.enable_grad if is_training else torch.no_grad
    with context():
        for i, batch in enumerate(tqdm(dataloader, leave=False)):
            total_loss, max_error, metrics, plot_data = process_batch(
                batch, model, mse_w, lncc_loss, lncc_w, reg_penalty, reg_w, save_warps, device
            )

            if is_training:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            metric_tracker.update(max_error, **metrics)

            if i % save_metrics_every == save_metrics_every - 1:
                metric_tracker.save_metrics(writer, epoch * dataloader_len + i)
                metric_tracker.print_running()
                metric_tracker.reset_running()
                save_plot(nrows=math.ceil(len(plot_data) / 4),
                          ncols=4,
                          mode=mode,
                          writer=writer,
                          step=epoch * dataloader_len + i,
                          local_save=local_plot_save,
                          **plot_data)

    metric_tracker.print_epoch()

    return total_loss.item() if mode == 'val' else None
