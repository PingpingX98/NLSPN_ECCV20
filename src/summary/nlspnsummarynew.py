from PIL import Image
from . import BaseSummary
import imageio
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import matplotlib.cm as mcm
import matplotlib.colors as colors

cmap = 'jet'
cm = plt.get_cmap(cmap)

class NLSPNSummarynew(BaseSummary):
    def __init__(self, log_dir, mode, args, loss_name, metric_name):
        assert mode in ['train', 'val', 'test'], \
            "mode should be one of ['train', 'val', 'test'] " \
            "but got {}".format(mode)

        super(NLSPNSummarynew, self).__init__(log_dir, mode, args)

        self.log_dir = log_dir
        self.mode = mode
        self.args = args

        self.loss = []
        self.metric = []

        self.loss_name = loss_name
        self.metric_name = metric_name

        self.path_output = None

        # ImageNet normalization
        self.img_mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        self.img_std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

    def update(self, global_step, sample, output):
        if self.loss_name is not None:
            self.loss = np.concatenate(self.loss, axis=0)
            self.loss = np.mean(self.loss, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Loss')]
            for idx, loss_type in enumerate(self.loss_name):
                val = self.loss[0, idx]
                self.add_scalar('Loss/' + loss_type, val, global_step)

                msg += ["{:<s}: {:.4f}  ".format(loss_type, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_loss = open(self.f_loss, 'a')
            f_loss.write('{:05d} | {}\n'.format(global_step, msg))
            f_loss.close()

        if self.metric_name is not None:
            self.metric = np.concatenate(self.metric, axis=0)
            self.metric = np.mean(self.metric, axis=0, keepdims=True)

            msg = [" {:<9s}|  ".format('Metric')]
            for idx, name in enumerate(self.metric_name):
                val = self.metric[0, idx]
                self.add_scalar('Metric/' + name, val, global_step)

                msg += ["{:<s}: {:.5f}  ".format(name, val)]

                if (idx + 1) % 10 == 0:
                    msg += ["\n             "]

            msg = "".join(msg)
            print(msg)

            f_metric = open(self.f_metric, 'a')
            f_metric.write('{:05d} | {}\n'.format(global_step, msg))
            f_metric.close()

        # Un-normalization
        rgb = sample['rgb'].detach()
        rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))
        rgb = rgb.data.cpu().numpy()

        dep = sample['dep'].detach().data.cpu().numpy()
        gt = sample['gt'].detach().data.cpu().numpy()
        pred = output['pred'].detach().data.cpu().numpy()

        if output['confidence'] is not None:
            confidence = output['confidence'].data.cpu().numpy()
        else:
            confidence = np.zeros_like(dep)

        num_summary = rgb.shape[0]
        if num_summary > self.args.num_summary:
            num_summary = self.args.num_summary

            rgb = rgb[0:num_summary, :, :, :]
            dep = dep[0:num_summary, :, :, :]
            gt = gt[0:num_summary, :, :, :]
            pred = pred[0:num_summary, :, :, :]
            confidence = confidence[0:num_summary, :, :, :]

        rgb = np.clip(rgb, a_min=0, a_max=1.0)
        dep = np.clip(dep, a_min=0, a_max=self.args.max_depth)
        gt = np.clip(gt, a_min=0, a_max=self.args.max_depth)
        pred = np.clip(pred, a_min=0, a_max=self.args.max_depth)
        confidence = np.clip(confidence, a_min=0, a_max=1.0)

        list_img = []

        for b in range(0, num_summary):
            rgb_tmp = rgb[b, :, :, :]
            dep_tmp = dep[b, 0, :, :]
            gt_tmp = gt[b, 0, :, :]
            pred_tmp = pred[b, 0, :, :] 
            confidence_tmp = confidence[b, 0, :, :]
            
            # norm = plt.Normalize(vmin=gt_tmp.min(), vmax=gt_tmp.max())

            dep_tmp = 255.0 * dep_tmp / self.args.max_depth
            gt_tmp = 255.0 * gt_tmp / self.args.max_depth
            pred_tmp = 255.0 * pred_tmp / self.args.max_depth
            confidence_tmp = 255.0 * confidence_tmp

            dep_tmp = cm(dep_tmp.astype('uint8'))
            gt_tmp = cm(gt_tmp.astype('uint8'))
            pred_tmp = cm(pred_tmp.astype('uint8'))
            confidence_tmp = cm(confidence_tmp.astype('uint8'))

            dep_tmp = np.transpose(dep_tmp[:, :, :3], (2, 0, 1))
            gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
            pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
            confidence_tmp = np.transpose(confidence_tmp[:, :, :3], (2, 0, 1))

            img = np.concatenate((rgb_tmp, dep_tmp, pred_tmp, gt_tmp,
                                  confidence_tmp), axis=1)

            list_img.append(img)

        img_total = np.concatenate(list_img, axis=2)
        img_total = torch.from_numpy(img_total)

        self.add_image(self.mode + '/images', img_total, global_step)

        self.add_scalar('Etc/gamma', output['gamma'], global_step)

        self.flush()

        # Reset
        self.loss = []
        self.metric = []

    def save(self, epoch, idx, sample, output, id_in_batch=0):
        with torch.no_grad():
            if self.args.save_result_only:
                # print("Saving only result...")
                self.path_output = '{}/{}/epoch{:04d}'.format(self.log_dir,
                                                              self.mode, epoch)
                
                os.makedirs(self.path_output, exist_ok=True)
                path_save_pred = os.path.join(self.path_output, 'depth')
                path_save_color = os.path.join(self.path_output, 'depthcolor')
                os.makedirs(path_save_pred, exist_ok=True)
                os.makedirs(path_save_color, exist_ok=True)
                path_save_pred = '{}/{:08d}.png'.format(path_save_pred, idx)
                path_save_color = '{}/{:08d}.png'.format(path_save_color, idx)
                
                pred = output['pred'].detach()

                pred = torch.clamp(pred, min=0)

                pred = pred[0, 0, :, :].data.cpu().numpy()

                pred = (pred*256.0).astype(np.uint16)
                color_depth = self.ColorizeNew(pred, norm_type='LogNorm', offset=1.)
                imageio.imwrite(path_save_pred, pred)
                imageio.imwrite(path_save_color, color_depth)
            else:
                rgb = sample['rgb'].detach()
                dep = sample['dep'].detach()
                pred = output['pred'].detach()
                gt = sample['gt'].detach()

                pred = torch.clamp(pred, min=0)
                
                # Un-normalization
                rgb.mul_(self.img_std.type_as(rgb)).add_(self.img_mean.type_as(rgb))
                
                rgb = rgb[0, :, :, :].data.cpu().numpy()
                dep = dep[0, 0, :, :].data.cpu().numpy()
                pred = pred[0, 0, :, :].data.cpu().numpy()
                # print("pred min:", pred.min(), "max:", pred.max(), "dtype:", pred.dtype)
                gt = gt[0, 0, :, :].data.cpu().numpy()
                rgb = np.transpose(rgb, (1, 2, 0))
                pred_uint16 = (pred * 256).astype(np.uint16)  # 若 pred 为0-255范围
                
                pred_final = pred
                

                norm = plt.Normalize(vmin=gt.min(), vmax=gt.max())
            
                self.path_output = '{}/{}/epoch{:04d}/{:08d}'.format(
                    self.log_dir, self.mode, epoch, idx)
                os.makedirs(self.path_output, exist_ok=True)
                
                path_save_rgb = '{}/01_rgb.png'.format(self.path_output)
                path_save_dep = '{}/02_dep.png'.format(self.path_output)
                path_save_pred = '{}/03_pred.png'.format(
                    self.path_output)
                path_save_pred_visual = '{}/04_pred_final.png'.format(
                    self.path_output)
                    
                # transfer to 16-bit
                pred = pred.astype(np.uint16)
                #imageio.imwrite(path_save_rgb, rgb)
                plt.imsave(path_save_rgb, rgb, cmap=cmap)
                plt.imsave(path_save_dep, cm(norm(dep)))
                # imageio.imwrite(path_save_pred, pred)
                imageio.imwrite(path_save_pred, pred_uint16, format='PNG')
                plt.imsave(path_save_pred_visual, cm(norm(pred_final)))
                           
    def ColorizeNew(self, depth, norm_type='LogNorm', offset=1.):
        cmap = mcm.jet
        depth = (depth - depth.min()) / (depth.max() - depth.min()) + offset
        Norm = getattr(colors, norm_type)
        norm = Norm(vmin=depth.min(), vmax=depth.max(), clip=True)
        m = mcm.ScalarMappable(norm=norm, cmap=cmap)
        depth_color = (255 * m.to_rgba(depth)[:, :, 0:3]).astype(np.uint8)
        return depth_color                   
    
    def log_time(self, epoch, t_total, t_avg,
                 t_bench_avg, t_bench_total):
        """ 
        # 1) TensorBoard
        self.add_scalar('Time/Total_elapsed',   t_total,        epoch)
        self.add_scalar('Time/Avg_traditional', t_avg,          epoch)
        self.add_scalar('Time/Avg_benchmark',   t_bench_avg,    epoch)
        self.add_scalar('Time/Total_benchmark', t_bench_total,  epoch) """

        with open(self.f_metric, 'a') as f:
            f.write(f'{epoch:05d} | '
                    f'Total_elapsed={t_total:.6f}  '
                    f'Avg_traditional={t_avg:.6f}  '
                    f'Avg_benchmark={t_bench_avg:.6f}  '
                    f'Total_benchmark={t_bench_total:.6f}\n')

        self.flush()
    
    def colorize(self, depth, gt, percentile=1):
        cmap = mcm.jet
        vmin = np.percentile(gt, percentile)
        vmax = gt.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        m = mcm.ScalarMappable(norm=norm, cmap=cmap)
        depth_color = (255 * m.to_rgba(depth)[:, :, :3]).astype(np.uint8)
        return depth_color