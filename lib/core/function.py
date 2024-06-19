import time
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)

# Variabili globali per salvare le probabilità e le etichette
global global_all_probs, global_all_targets
global_all_probs = []
global_all_targets = []

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    scale_factor = config.LOSS.SCALE_FACTOR

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sr_losses = AverageMeter()  # For tracking super-resolution loss
    total_losses = AverageMeter()  # For tracking total loss
    acc = AverageMeter()

    model.train()

    print("------------- INIZIO ALLENAMENTO -------------")

    end = time.time()
    for i, (input, target, target_weight, meta, y_sr_target) in enumerate(train_loader):
       
        data_time.update(time.time() - end)

        # Compute output and super-resolution output
        output, y_sr = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        y_sr_target = y_sr_target.cuda(non_blocking=True)

        # Ensure y_sr has 3 channels if needed
        if y_sr.shape[1] != 3:
            y_sr = torch.nn.Conv2d(y_sr.shape[1], 3, kernel_size=1).cuda()(y_sr)

        # Resize y_sr to match y_sr_target size without normalizing
        y_sr = F.interpolate(y_sr, size=y_sr_target.shape[2:], mode='bilinear', align_corners=False)

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
        total_losses.update(total_loss.item(), input.size(0))  # Update total loss

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
                  'Total Loss {total_loss.val:.5f} ({total_loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, sr_loss=sr_losses, total_loss=total_losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_sr_loss', sr_losses.val, global_steps)  # Log super-resolution loss
            writer.add_scalar('train_total_loss', total_losses.val, global_steps)  # Log total loss
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output, prefix)

def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    global global_all_probs, global_all_targets

    scale_factor = config.LOSS.SCALE_FACTOR

    batch_time = AverageMeter()
    losses = AverageMeter()
    sr_losses = AverageMeter()
    total_losses = AverageMeter()
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

    print("------------- INIZIO TEST -------------")

    # Liste per salvare le probabilità e le etichette vere
    all_probs = []
    all_targets = []

    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta, y_sr_target) in enumerate(val_loader):
            outputs, y_sr = model(input)

            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped, _ = model(input_flipped)
                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped
                output_flipped = flip_back(output_flipped.cpu().numpy(), val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]
                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            y_sr_target = y_sr_target.cuda(non_blocking=True)

            # Ensure y_sr has 3 channels if needed
            if y_sr.shape[1] != 3:
              y_sr = torch.nn.Conv2d(y_sr.shape[1], 3, kernel_size=1).cuda()(y_sr)
          
            # Resize y_sr to match y_sr_target size without normalizing
            y_sr = F.interpolate(y_sr, size=y_sr_target.shape[2:], mode='bilinear', align_corners=False)

            loss = criterion(output, target, target_weight)
            sr_loss = F.mse_loss(y_sr, y_sr_target)
            scaled_sr_loss = sr_loss * scale_factor
            total_loss = loss + scaled_sr_loss

            num_images = input.size(0)
            losses.update(loss.item(), num_images)
            sr_losses.update(scaled_sr_loss.item(), num_images)
            total_losses.update(total_loss.item(), num_images)

            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())
            acc.update(avg_acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

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

            # Salva le probabilità (predizioni) e le etichette vere
            all_probs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'SR Loss {sr_loss.val:.4f} ({sr_loss.avg:.4f})\t' \
                      'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, sr_loss=sr_losses, total_loss=total_losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, target, pred*4, output, prefix)

        name_values, perf_indicator = val_dataset.evaluate(config, all_preds, output_dir, all_boxes, image_path, filenames, imgnums)

        # Converti le liste in array numpy senza appiattirle
        all_probs = np.concatenate(all_probs, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()

        # Genera etichette binarie basate su una soglia della media del GT
        mean_target = np.mean(all_targets)
        binary_labels = (all_targets > mean_target).astype(int)   # Etichetta 1 se l'errore è inferiore alla media, altrimenti 0

        # Aggiungi le probabilità e le etichette globali
        global_all_probs.extend(all_probs)
        global_all_targets.extend(binary_labels)

        model_name = config.MODEL.NAME  # Assicurati di definire model_name

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
