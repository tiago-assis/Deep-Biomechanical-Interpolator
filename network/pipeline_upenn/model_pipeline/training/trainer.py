from tqdm import tqdm
import torch
from model_pipeline.metrics.losses import Jacobian_det, Hessian_penalty
from model_pipeline.metrics.logger import save_plot
from model_pipeline.utils.utils import warp_tensor


def train(train_loader, 
          model, 
          mse_loss, 
          optimizer, 
          epoch, 
          writer, 
          metric_tracker,
          mse_w=1.0, 
          lncc_loss=None, 
          lncc_w=1.0,
          hessian=False, 
          jdet=True, 
          reg_w=1.0, 
          save_metrics_every=20, 
          #save_training_plots=False, 
          #local_plot_save=False, 
          device='cuda'):
    model.train()
    
    metric_tracker.reset_running()
    metric_tracker.reset_epoch()
    
    for i, batch in enumerate(tqdm(train_loader, leave=False)):
        img, gt_ddf, init_ddf, brain_seg, tumor_seg, upenn_edema_seg = batch.values()
        img = img.to(device)
        gt_ddf = gt_ddf.to(device)
        init_ddf = init_ddf.to(device)
        tumor_seg = tumor_seg.to(device)
        upenn_edema_seg = upenn_edema_seg.to(device)
        upenn_edema_seg = upenn_edema_seg > 0
        brain_seg = brain_seg.to(device)
        brain_seg = brain_seg > 0
        
        inputs = torch.cat([img, init_ddf], dim=1)
        pred_ddf = model(inputs, init_ddf)

        mse = mse_loss(torch.where(brain_seg, pred_ddf, 0.0), torch.where(brain_seg, gt_ddf, 0.0)) * mse_w
        if lncc_loss is not None:
            lncc = lncc_loss(warp_tensor(img, pred_ddf), warp_tensor(img, gt_ddf)) * lncc_w
        else:
            lncc = 0.0

        if jdet:
            reg = Jacobian_det(torch.where(brain_seg, pred_ddf, 0.0), tumor_seg) * reg_w
        elif hessian:
            reg = Hessian_penalty(torch.where(brain_seg, pred_ddf, 0.0)) * reg_w
        else:
            reg = 0.0
        
        total_loss = mse + lncc + reg
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if lncc_loss is not None:
            metrics = {
                'Initial MSE': mse_loss(torch.where(brain_seg, init_ddf, 0.0), torch.where(brain_seg, gt_ddf, 0.0)).item() * mse_w,
                'Initial MSE (masked)': mse_loss(torch.where(upenn_edema_seg, init_ddf, 0.0), torch.where(upenn_edema_seg, gt_ddf, 0.0)).item() * mse_w,
                'Initial LNCC': lncc_loss(warp_tensor(img, init_ddf), warp_tensor(img, gt_ddf)).item() * lncc_w,
                'Prediction MSE': mse.item(),
                'Prediction MSE (masked)': mse_loss(torch.where(upenn_edema_seg, pred_ddf, 0.0), torch.where(upenn_edema_seg, gt_ddf, 0.0)).item() * mse_w,
                'Prediction LNCC': lncc.item(),
                'Regularizer': reg.item(),
                'Regularizer (masked)': Jacobian_det(torch.where(upenn_edema_seg, pred_ddf, 0.0), tumor_seg).item() * reg_w if jdet else Hessian_penalty(torch.where(upenn_edema_seg, pred_ddf, 0.0)).item() * reg_w,
                'Total Loss': total_loss.item()
            }
        else:
            metrics = {
                'Initial MSE': mse_loss(torch.where(brain_seg, init_ddf, 0.0), torch.where(brain_seg, gt_ddf, 0.0)).item() * mse_w,
                'Initial MSE (masked)': mse_loss(torch.where(upenn_edema_seg, init_ddf, 0.0), torch.where(upenn_edema_seg, gt_ddf, 0.0)).item() * mse_w,
                'Prediction MSE': mse.item(),
                'Prediction MSE (masked)': mse_loss(torch.where(upenn_edema_seg, pred_ddf, 0.0), torch.where(upenn_edema_seg, gt_ddf, 0.0)).item() * mse_w,
                'Regularizer': reg.item(),
                'Regularizer (masked)': Jacobian_det(torch.where(upenn_edema_seg, pred_ddf, 0.0), tumor_seg).item() * reg_w if jdet else Hessian_penalty(torch.where(upenn_edema_seg, pred_ddf, 0.0)).item() * reg_w,
                'Total Loss': total_loss.item()
            }
        max_error = torch.round(torch.norm(torch.where(brain_seg, pred_ddf, 0.0) - torch.where(brain_seg, gt_ddf, 0.0), p=1, dim=1).max(), decimals=4)
        metric_tracker.update(max_error, **metrics)
        
        ########
        metric_tracker.print_running()

        if i % save_metrics_every == save_metrics_every - 1:
            metric_tracker.save_metrics(writer, epoch * len(train_loader) + i)
            metric_tracker.reset_running()

            #if save_training_plots:
            #    plot_data = {
            #        "MRI Scan": img,
            #        "Pred Disp Field": pred_ddf,
            #        "GT Disp Field (Sim)": gt_ddf,
            #        "Init Disp Field (Interp)": init_ddf,
            #        "Brain Mask": brain_seg,
            #        "Tumor Mask": tumor_seg
            #    }
            #    save_plot(nrows=2, ncols=4, writer=writer, step=epoch * len(train_loader) + i, local_save=local_plot_save, **plot_data)
        
    metric_tracker.print_epoch()

def evaluate(val_loader,
             model,
             mse_loss, 
             epoch, 
             writer, 
             metric_tracker,
             mse_w=1.0, 
             lncc_loss=None, 
             lncc_w=1.0,
             hessian=False, 
             jdet=True, 
             reg_w=1.0, 
             save_metrics_every=4, 
             local_plot_save=False, 
             device='cuda'):
    model.eval()

    metric_tracker.reset_running()
    metric_tracker.reset_epoch()
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, leave=False)):
            img, gt_ddf, init_ddf, brain_seg, tumor_seg, upenn_edema_seg = batch.values()
            img = img.to(device)
            gt_ddf = gt_ddf.to(device)
            init_ddf = init_ddf.to(device)
            upenn_edema_seg = upenn_edema_seg.to(device)
            upenn_edema_seg = upenn_edema_seg > 0
            tumor_seg = tumor_seg.to(device)
            brain_seg = brain_seg.to(device)
            brain_seg = brain_seg > 0
            
            inputs = torch.cat([img, init_ddf], dim=1)
            pred_ddf = model(inputs, init_ddf)
            
            mse = mse_loss(torch.where(brain_seg, pred_ddf, 0.0), torch.where(brain_seg, gt_ddf, 0.0)) * mse_w
            if lncc_loss is not None:
                lncc = lncc_loss(warp_tensor(img, pred_ddf), warp_tensor(img, gt_ddf)) * lncc_w
            else:
                lncc = 0.0

            if jdet:
                reg = Jacobian_det(torch.where(brain_seg, pred_ddf, 0.0), tumor_seg) * reg_w
            elif hessian:
                reg = Hessian_penalty(torch.where(brain_seg, pred_ddf, 0.0)) * reg_w
            else:
                reg = 0.0
            
            total_loss = mse + lncc + reg

            if lncc_loss is not None:
                print(lncc_loss)
                metrics = {
                    'Initial MSE': mse_loss(torch.where(brain_seg, init_ddf, 0.0), torch.where(brain_seg, gt_ddf, 0.0)).item(),
                    'Initial MSE (masked)': mse_loss(torch.where(upenn_edema_seg, init_ddf, 0.0), torch.where(upenn_edema_seg, gt_ddf, 0.0)).item(),
                    'Initial LNCC': lncc_loss(warp_tensor(img, init_ddf), warp_tensor(img, gt_ddf)).item(),
                    'Prediction MSE': mse.item(),
                    'Prediction MSE (masked)': mse_loss(torch.where(upenn_edema_seg, pred_ddf, 0.0), torch.where(upenn_edema_seg, gt_ddf, 0.0)).item(),
                    'Prediction LNCC': lncc.item(),
                    'Regularizer': reg.item(),
                    'Regularizer (masked)': Jacobian_det(torch.where(upenn_edema_seg, pred_ddf, 0.0), tumor_seg).item() if jdet else Hessian_penalty(torch.where(upenn_edema_seg, pred_ddf, 0.0)).item(),
                    'Total Loss': total_loss.item()
                }
            else:
                metrics = {
                    'Initial MSE': mse_loss(torch.where(brain_seg, init_ddf, 0.0), torch.where(brain_seg, gt_ddf, 0.0)).item(),
                    'Initial MSE (masked)': mse_loss(torch.where(upenn_edema_seg, init_ddf, 0.0), torch.where(upenn_edema_seg, gt_ddf, 0.0)).item(),
                    'Prediction MSE': mse.item(),
                    'Prediction MSE (masked)': mse_loss(torch.where(upenn_edema_seg, pred_ddf, 0.0), torch.where(upenn_edema_seg, gt_ddf, 0.0)).item(),
                    'Regularizer': reg.item(),
                    'Regularizer (masked)': Jacobian_det(torch.where(upenn_edema_seg, pred_ddf, 0.0), tumor_seg).item() if jdet else Hessian_penalty(torch.where(upenn_edema_seg, pred_ddf, 0.0)).item(),
                    'Total Loss': total_loss.item()
                }
                max_error = torch.norm(torch.where(brain_seg, pred_ddf, 0.0) - torch.where(brain_seg, gt_ddf, 0.0), p=1, dim=1).max()
                metric_tracker.update(max_error, **metrics)

                if i % save_metrics_every == save_metrics_every - 1:
                    metric_tracker.save_metrics(writer, epoch * len(val_loader) + i)
                    metric_tracker.reset_running()

                    plot_data = {
                        "MRI Scan": img,
                        "Pred Disp Field": pred_ddf,
                        "GT Disp Field (Sim)": gt_ddf,
                        "Init Disp Field (Interp)": init_ddf,
                        "Brain Mask": brain_seg,
                        "Tumor Mask": tumor_seg,
                        "Tumor+Edema Mask": upenn_edema_seg
                    }
                    save_plot(nrows=2, ncols=4, writer=writer, step=epoch * len(val_loader) + i, local_save=local_plot_save, **plot_data)
                
    metric_tracker.print_epoch()
