import time
import logging
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    scale_factor = config.LOSS.SCALE_FACTOR

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sr_losses = AverageMeter()  # For tracking super-resolution loss
    acc = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta, y_sr_target) in enumerate(train_loader):
        if i >= 100:  # Limit to 100 samples per epoch
            break

        data_time.update(time.time() - end)

        # Compute output and super-resolution output
        output, y_sr = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        y_sr_target = y_sr_target.cuda(non_blocking=True)

        # Ensure y_sr has 3 channels
        y_sr = F.interpolate(y_sr, size=y_sr_target.shape[2:], mode='bilinear', align_corners=False)
        y_sr = torch.nn.Conv2d(y_sr.shape[1], 3, kernel_size=1).cuda()(y_sr)

        if isinstance(output, list):
            loss = criterion(output[0], target, target_weight)
            for output_i in output[1:]:
                loss += criterion(output_i, target, target_weight)
        else:
            loss = criterion(output, target, target_weight)

        # Calculate super-resolution loss
        sr_loss = F.mse_loss(y_sr, y_sr_target)

        # Apply scale factor to sr_loss
        scaled_sr_loss = sr_loss * scale_factor
        total_loss = loss + scaled_sr_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        losses.update(loss.item(), input.size(0))
        sr_losses.update(scaled_sr_loss.item(), input.size(0))  # Update super-resolution loss

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'SR Loss {sr_loss.val:.5f} ({sr_loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, sr_loss=sr_losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_sr_loss', sr_losses.val, global_steps)  # Log super-resolution loss
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    scale_factor = config.LOSS.SCALE_FACTOR

    batch_time = AverageMeter()
    losses = AverageMeter()
    sr_losses = AverageMeter()  # For tracking super-resolution loss
    total_losses = AverageMeter()  # For tracking total loss
    acc = AverageMeter()

    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0

    all_targets = []
    all_losses = []
    all_total_losses = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta, y_sr_target) in enumerate(val_loader):
            if i >= 50:  # Limit to 50 samples per validation
                break

            output, y_sr = model(input)

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            y_sr_target = y_sr_target.cuda(non_blocking=True)

            # Ensure y_sr has 3 channels
            y_sr = F.interpolate(y_sr, size=y_sr_target.shape[2:], mode='bilinear', align_corners=False)
            y_sr = torch.nn.Conv2d(y_sr.shape[1], 3, kernel_size=1).cuda()(y_sr)

            if isinstance(output, list):
                loss = criterion(output[0], target, target_weight)
                for output_i in output[1:]:
                    loss += criterion(output_i, target, target_weight)
            else:
                loss = criterion(output, target, target_weight)

            # Calculate super-resolution loss
            sr_loss = F.mse_loss(y_sr, y_sr_target)
            scaled_sr_loss = sr_loss * scale_factor
            total_loss = loss + scaled_sr_loss

            num_images = input.size(0)
            losses.update(loss.item(), num_images)
            sr_losses.update(scaled_sr_loss.item(), num_images)  # Update super-resolution loss
            total_losses.update(total_loss.item(), num_images)

            all_targets.append(target.cpu().numpy())
            all_losses.append(loss.item())
            all_total_losses.append(total_loss.item())

            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())
            acc.update(avg_acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'SR Loss {sr_loss.val:.4f} ({sr_loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, sr_loss=sr_losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                  prefix)

        all_targets = np.concatenate(all_targets).ravel().tolist()
        all_losses = np.array(all_losses).tolist()
        all_total_losses = np.array(all_total_losses).tolist()

        data_to_save = {
            'targets': all_targets,
            'losses': all_losses,
            'total_losses': all_total_losses
        }

        with open(os.path.join(output_dir, 'roc_data.json'), 'w') as f:
            json.dump(data_to_save, f)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

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
            writer.add_scalar('valid_sr_loss', sr_losses.avg, global_steps)  # Log super-resolution loss
            writer.add_scalar('valid_acc', acc.avg, global_steps)

            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), global_steps)
            else:
                writer.add_scalars('valid', dict(name_values), global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator

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
