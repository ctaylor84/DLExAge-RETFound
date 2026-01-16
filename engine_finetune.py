# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import csv
import math
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

import util.lr_sched as lr_sched
import util.misc as misc


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    norm_params,
    args,
    max_norm: float = 0,
    log_writer=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.float, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples).squeeze()
            loss = criterion(
                outputs, (targets - norm_params["mean"]) / norm_params["std"]
            )

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def true_vs_pred_boxplot(plot_path, target_list, prediction_list):
    value_max = round(np.amax(target_list))
    value_min = round(np.amin(target_list))
    total_bins = value_max - value_min
    bins = np.arange(value_min, value_max + 2, step=1.0) - 0.5
    digitized = np.digitize(target_list, bins)
    residuals_list = target_list - prediction_list

    mean_x = np.arange(value_min, value_max - 1)
    bin_loss_means = [
        np.mean(residuals_list[digitized == i]) for i in range(1, total_bins)
    ]
    bin_pred_means = [
        np.mean(prediction_list[digitized == i]) for i in range(1, total_bins)
    ]
    box_bins_loss = [residuals_list[digitized == i] for i in range(1, total_bins)]
    box_bins_pred = [prediction_list[digitized == i] for i in range(1, total_bins)]
    box_bins_counts = [
        target_list[digitized == i].shape[0] for i in range(1, total_bins)
    ]
    box_bin_x = list(range(1, total_bins))

    box_labels = ["" if int(x) % 5 != 0 else str(x) for x in mean_x]
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.boxplot(
        box_bins_loss,
        positions=box_bin_x,
        widths=0.5,
        flierprops=dict({"markersize": 3}),
    )
    ax.set_xticks(ticks=box_bin_x)
    ax.set_xticklabels(labels=box_labels)
    ax.plot(box_bin_x, bin_loss_means, c="blue")
    ax.plot(box_bin_x, np.zeros(len(box_bin_x)), c="red")
    ax.set_xlabel("True value (years)")
    ax.set_ylabel("Residual (years)")
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax2 = ax.twinx()
    ax2.plot(box_bin_x, box_bins_counts, c="green")
    ax2.set_ylabel("Number of samples")
    ax2.yaxis.get_ticklocs(minor=True)
    ax2.minorticks_on()
    ax2.tick_params(axis="x", which="minor", bottom=False)
    plt.tight_layout()
    plt.savefig(plot_path + "residuals.jpg", dpi=600)
    plt.close(fig)

    plt.figure(figsize=(12, 8))
    plt.boxplot(
        box_bins_pred,
        positions=box_bin_x,
        widths=0.5,
        flierprops=dict({"markersize": 3}),
    )
    plt.xticks(ticks=box_bin_x, labels=box_labels)
    plt.plot(box_bin_x, mean_x, c="red")
    plt.plot(box_bin_x, bin_pred_means, c="blue")
    plt.xlabel("True value (years)")
    plt.ylabel("Predicted value (years)")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(plot_path + "true_vs_pred.jpg", dpi=600)
    plt.close()


def true_vs_pred_scatter(plot_path, target_list, prediction_list):
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(target_list, prediction_list, s=1)
    plt.plot(target_list, target_list, c="red")
    plt.xlabel("True value (years)")
    plt.ylabel("Predicted value (years)")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(plot_path + "true_vs_pred_scatter.jpg", dpi=600)
    plt.close(fig)


@torch.no_grad()
def evaluate(data_loader, model, criterion, norm_params, device, task, epoch, mode):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    if not os.path.exists(task):
        os.makedirs(task)

    prediction_list = list()
    target_list = list()

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images).squeeze(dim=1)
            loss = criterion(
                output, (target - norm_params["mean"]) / norm_params["std"]
            )
            output_np = (
                ((output * norm_params["std"]) + norm_params["mean"])
                .cpu()
                .detach()
                .numpy()
            )
            target_np = target.cpu().detach().numpy()
            prediction_list.append(output_np)
            target_list.append(target_np)

        batch_rmse = root_mean_squared_error(target_np, output_np)
        batch_r2 = r2_score(target_np, output_np)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["rmse"].update(batch_rmse, n=batch_size)
        metric_logger.meters["r2"].update(batch_r2, n=batch_size)

    prediction_list = np.concatenate(prediction_list)
    target_list = np.concatenate(target_list)

    rmse = root_mean_squared_error(target_list, prediction_list)
    mse = mean_squared_error(target_list, prediction_list)
    mae = mean_absolute_error(target_list, prediction_list)
    r2 = r2_score(target_list, prediction_list)

    metric_logger.synchronize_between_processes()

    print(
        "Sklearn Metrics - RMSE: {:.4f} MSE: {:.4f} MAE: {:.4f} r2: {:.4f}".format(
            rmse, mse, mae, r2
        )
    )
    results_path = task + "_metrics_{}.csv".format(mode)
    with open(results_path, mode="a", newline="", encoding="utf8") as cfa:
        wf = csv.writer(cfa)
        data2 = [[rmse, mse, mae, r2, metric_logger.loss]]
        for i in data2:
            wf.writerow(i)

    if mode == "test":
        true_vs_pred_boxplot(task + "test_", target_list, prediction_list)
        true_vs_pred_scatter(task + "test_", target_list, prediction_list)
        np.save(task + "test_pred", prediction_list)
    elif mode == "val_final":
        np.save(task + "val_pred", prediction_list)
    elif mode == "train_eval":
        np.save(task + "train_pred", prediction_list)
    elif epoch % 5 == 0:
        true_vs_pred_boxplot(task + str(epoch) + "_", target_list, prediction_list)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, mse
