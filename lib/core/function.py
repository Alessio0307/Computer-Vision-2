# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
 
from tqdm import tqdm
 
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
 
logger = logging.getLogger(__name__)

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    scale_factor = config.LOSS.SCALE_FACTOR

    losses = AverageMeter()
    sr_losses = AverageMeter()
    total_losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    print("\n---- Training ----")
    with tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}") as tepoch:
        for i, (input, target, target_weight, meta, y_sr_target) in enumerate(tepoch):
        
            outputs = model(input.cuda())

            if isinstance(outputs, tuple):
                output, y_sr = outputs
            else:
                output = outputs
                y_sr = None

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            if y_sr is not None:
                y_sr_target = y_sr_target.cuda(non_blocking=True)

            if isinstance(output, list):
                loss = criterion(output[0], target, target_weight)
                for output_i in output[1:]:
                    loss += criterion(output_i, target, target_weight)
            else:
                loss = criterion(output, target, target_weight)

            if y_sr is not None:
                sr_loss = F.mse_loss(y_sr, y_sr_target)
                scaled_sr_loss = sr_loss * scale_factor
                total_loss = loss + scaled_sr_loss
                sr_losses.update(sr_loss.item(), input.size(0))
            else:
                total_loss = loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            losses.update(loss.item(), input.size(0))
            total_losses.update(total_loss.item(), input.size(0))

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            tepoch.set_postfix_str(f"total_loss: {total_losses.avg*100:.4f} HR_loss: {losses.avg*100:.4f} SR_loss: {sr_losses.avg*100:.4f} accuracy: {acc.avg:.4f}")

            if i % config.PRINT_FREQ == 0:
                prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                save_debug_images(config, input, meta, target, pred*4, output, prefix)
    
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    writer.add_scalar('train_loss', losses.val, global_steps)
    writer.add_scalar('train_sr_loss', sr_losses.val, global_steps)
    writer.add_scalar('train_total_loss', total_losses.val, global_steps)
    writer.add_scalar('train_acc', acc.val, global_steps)
    writer_dict['train_global_steps'] = global_steps + 1

    # Return the metrics as a dictionary
    metr_dict = {
        "loss": losses.avg,
        "sr_loss": sr_losses.avg,
        "total_loss": total_losses.avg,
        "accuracy": acc.avg,
    }

    return metr_dict

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    scale_factor = config.LOSS.SCALE_FACTOR

    losses = AverageMeter()
    sr_losses = AverageMeter()
    total_losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    print("---- Validation ----")
    with torch.no_grad():
        with tqdm(val_loader, total=len(val_loader), desc="Validation") as tepoch:
            for i, (input, target, target_weight, meta, y_sr_target) in enumerate(tepoch):
            
                outputss = model(input.cuda())

                if isinstance(outputss, tuple):
                    outputs, y_sr = outputss
                else:
                    outputs = outputss
                    y_sr = None

                if isinstance(outputs, list):
                    output = outputs[-1]
                else:
                    output = outputs

                if config.TEST.FLIP_TEST:
                    input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                    input_flipped = torch.from_numpy(input_flipped).cuda()
                    outputs_flipped = model(input_flipped)

                    if isinstance(outputs_flipped, list):
                        output_flipped = outputs_flipped[-1]
                    else:
                        output_flipped = outputs_flipped

                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                            val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]

                    output = (output + output_flipped) * 0.5

                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
                if y_sr is not None:
                    y_sr_target = y_sr_target.cuda(non_blocking=True)

                loss = criterion(output, target, target_weight)

                num_images = input.size(0)

                if y_sr is not None:
                    sr_loss = F.mse_loss(y_sr, y_sr_target)
                    scaled_sr_loss = sr_loss * scale_factor
                    total_loss = loss + scaled_sr_loss
                    sr_losses.update(sr_loss.item(), num_images)
                else:
                    total_loss = loss

                losses.update(loss.item(), num_images)
                total_losses.update(total_loss.item(), num_images)

                _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())
                acc.update(avg_acc, cnt)

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()

                preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])

                idx += num_images
                
                tepoch.set_postfix_str(f"total_loss: {total_losses.avg*100:.5f} HR_loss: {losses.avg*100:.5f} SR_loss: {sr_losses.avg*100:.5f} accuracy: {acc.avg:.3f}")
                if i % config.PRINT_FREQ == 0:
                    prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                    save_debug_images(config, input, meta, target, pred*4, output, prefix)

        name_values, perf_indicator = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, image_path, filenames, imgnums)
        model_name = config.MODEL.NAME  
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, global_steps)
            writer.add_scalar('valid_sr_loss', sr_losses.avg, global_steps)
            writer.add_scalar('valid_total_loss', total_losses.avg, global_steps)
            writer.add_scalar('valid_acc', acc.avg, global_steps)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    metr_dict = {
        "loss": losses.avg,
        "sr_loss": sr_losses.avg,
        "total_loss": total_losses.avg,
        "accuracy": acc.avg,
    }

    return perf_indicator, metr_dict

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
 
    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )
 
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
 
 
