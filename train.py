import torch
from tqdm import tqdm
import numpy as np
import torchmetrics
from utils import AvgMeter, get_lr


def train_epoch(model, train_loader, optimizer, criterion, lr_scheduler, metrics, device):
    """Train the model for one epoch."""
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for image, label in tqdm_object:
        image, label = image.to(device), label.to(device)
        prediction = model(image)
        loss = criterion(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        loss_meter.update(loss.item())
        for metric in metrics.values():
            metric.update(prediction, label)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    metric_results = {name: metric.compute().item() for name, metric in metrics.items()}
    for metric in metrics.values():
        metric.reset()

    return loss_meter, metric_results


def valid_epoch(model, valid_loader, criterion, metrics, device, mode='val', log_path=None, cm_file=None):
    """Validate or test the model."""
    loss_meter = AvgMeter()
    predictions_list = []
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    num_classes = next(iter(metrics.values())).num_classes
    confusion_matrix_metric = torchmetrics.ConfusionMatrix(num_classes=num_classes, task='multiclass').to(device)

    for image, label in tqdm_object:
        image, label = image.to(device), label.to(device)
        prediction = model(image)
        loss = criterion(prediction, label)
        loss_meter.update(loss.item())

        for metric in metrics.values():
            metric.update(prediction, label)
        if mode == 'test':
            confusion_matrix_metric.update(prediction.argmax(dim=1), label)
            predictions_list.append(prediction.detach().cpu().numpy())

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)

    metric_results = {f"{mode} {name}": metric.compute().item() for name, metric in metrics.items()}
    metric_results['loss'] = loss_meter.avg

    for metric in metrics.values():
        metric.reset()

    if mode == 'test' and log_path and cm_file:
        predictions_array = np.concatenate(predictions_list, axis=0)
        np.save(f"{log_path}/{mode}_predictions.npy", predictions_array)
        confusion_matrix = confusion_matrix_metric.compute()
        with open(f"{log_path}/{cm_file}", 'a') as f:
            f.write('Confusion Matrix:\n')
            f.write(str(confusion_matrix.cpu().numpy()) + '\n')
        confusion_matrix_metric.reset()

    return loss_meter, metric_results