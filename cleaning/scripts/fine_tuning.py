"""
Fine-tuning script for cleaning UNet models.

This script provides fine-tuning capabilities for UNet-based image cleaning models
using synthetic data with optional pre-trained model loading.
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from cleaning.models.SmallUnet.unet import SmallUnet
from cleaning.models.Unet.unet_model import UNet
from cleaning.utils.dataloader import MakeDataSynt
from cleaning.utils.loss import CleaningLoss
from util_files.metrics.raster_metrics import iou_score

MODEL_LOSS: Dict[str, Dict[str, torch.nn.Module]] = {
    "UNET": {
        "model": UNet(n_channels=3, n_classes=1, final_tanh=True),
        "loss": CleaningLoss(kind="BCE", with_restore=False),
    },
    "SmallUNET": {"model": SmallUnet(), "loss": CleaningLoss(kind="BCE", with_restore=False)},
    "UNET_MSE": {"model": UNet(n_channels=3, n_classes=1), "loss": CleaningLoss(kind="MSE", with_restore=False)},
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for fine-tuning script."""
    parser = argparse.ArgumentParser(description="Fine-tune cleaning UNet models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Model type: one of ["UNET", "UNET_MSE", "SmallUNET"]',
        choices=["UNET", "UNET_MSE", "SmallUNET"],
        default="UNET",
    )
    parser.add_argument("--n_epochs", type=int, help="Number of epochs for training", default=10)
    parser.add_argument("--datadir", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--valdatadir", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument(
        "--added_part", type=str, default="No", choices=["Yes", "No"], help="Use added part training mode"
    )

    return parser.parse_args()


def get_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    train_transform = transforms.Compose([transforms.ToTensor()])
    # TODO Make sure that this should be MakeDataSynt and not MakeData from dataloader.py
    dset_synt = MakeDataSynt(args.datadir, args.datadir, train_transform, 1)
    dset_val_synt = MakeDataSynt(args.valdatadir, args.valdatadir, train_transform)

    print(f"Batch size: {args.batch_size}")

    train_loader = DataLoader(
        dset_synt, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False  # 1 for CUDA
    )

    val_loader = DataLoader(dset_val_synt, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)  # 1 for CUDA

    return train_loader, val_loader


def validate(
    tb: SummaryWriter,
    val_loader: DataLoader,
    model: torch.nn.Module,
    loss_func: torch.nn.Module,
    global_step: int,
    added_part: bool = False,
) -> None:
    """Run validation and log metrics to TensorBoard."""
    val_loss_epoch = []
    val_iou_extract = []

    for x_input, y_extract, y_restor in tqdm(val_loader):
        x_input = torch.FloatTensor(x_input).cuda()
        y_extract = y_extract.type(torch.FloatTensor).cuda()

        logits_restor, logits_extract = None, model(x_input)  # restoration + extraction

        if added_part:
            # training "Unet on without filling wholes on h_gt in synthetic
            loss = loss_func(logits_extract, logits_restor, y_extract, y_restor)
            iou_scr = iou_score(
                torch.round(torch.clamp(logits_extract, 0, 1).cpu()).long().numpy(), y_extract.cpu().long().numpy()
            )
        else:
            # training "Unet on whole image nh_gt, default this one
            loss = loss_func(logits_extract, logits_restor, y_restor, y_extract)
            iou_scr = iou_score(
                torch.round(torch.clamp(logits_extract, 0, 1).cpu()).long().numpy(), y_restor.cpu().long().numpy()
            )

        val_iou_extract.append(iou_scr)
        val_loss_epoch.append(loss.cpu().data.numpy())
        del loss

    tb.add_scalar("val_loss", np.mean(val_loss_epoch), global_step=global_step)
    tb.add_scalar("val_iou_extract", np.mean(val_iou_extract), global_step=global_step)

    out_grid = torchvision.utils.make_grid(logits_extract.unsqueeze(1).cpu())
    input_grid = torchvision.utils.make_grid(x_input.cpu())
    tb.add_image(tag="val_out_extract", img_tensor=out_grid, global_step=global_step)
    tb.add_image(tag="val_input", img_tensor=input_grid, global_step=global_step)


def save_model(model: torch.nn.Module, path: str) -> None:
    """Save model to specified path."""
    print(f'Saving model to "{path}"')
    torch.save(model, path)


def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """Load model from specified path."""
    print(f'Loading model from "{path}"')
    model = torch.load(path)
    return model


def main(args: argparse.Namespace) -> None:
    """Main training function for fine-tuning cleaning models."""
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU training.")

    tb_dir = f"/logs/tb_logs_article/fine_tuning_{args.name}"
    tb = SummaryWriter(tb_dir)

    train_loader, val_loader = get_dataloaders(args)

    if args.model not in MODEL_LOSS:
        available_models = list(MODEL_LOSS.keys())
        raise ValueError(f"Unsupported model type '{args.model}'. Choose from: {available_models}")

    model = MODEL_LOSS[args.model]["model"]
    loss_func = MODEL_LOSS[args.model]["loss"]

    model = model.cuda()

    if args.model_path is not None:
        model = load_model(model, args.model_path)
    model.eval()
    tb.add_text(tag="model", text_string=repr(model))

    opt = torch.optim.Adam(model.parameters(), lr=0.0005)

    global_step = 0
    added_part = args.added_part == "Yes"

    for epoch in range(args.n_epochs):
        model.train()
        for x_input, y_extract, y_restor in tqdm(train_loader):
            x_input = torch.FloatTensor(x_input).cuda()
            y_extract = y_extract.type(torch.FloatTensor).cuda()

            logits_restor, logits_extract = None, model(x_input)  # restoration + extraction
            # training "Unet on whole image, aka the final result, default this one, make sure this image is in y_restor
            # or swap  y_extract,y_restor
            loss = loss_func(logits_extract, logits_restor, y_restor, y_extract)
            loss.backward()
            opt.step()
            opt.zero_grad()

            tb.add_scalar("train_loss", loss.cpu().data.numpy(), global_step=global_step)

            if global_step % 500 == 0:
                out_grid = torchvision.utils.make_grid(logits_extract.unsqueeze(1).cpu())
                input_grid = torchvision.utils.make_grid(x_input.cpu())

                tb.add_image(tag="train_out_extract", img_tensor=out_grid, global_step=global_step)
                tb.add_image(tag="train_input", img_tensor=input_grid, global_step=global_step)

                model.eval()
                with torch.no_grad():
                    validate(tb, val_loader, model, loss_func, global_step=global_step, added_part=added_part)
                model.train()
                save_model(model, os.path.join(tb_dir, f"model_it_{global_step}.pth"))

            del logits_extract
            del logits_restor

            global_step += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
