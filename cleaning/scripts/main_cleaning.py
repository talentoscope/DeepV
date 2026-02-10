"""
Main training script for DeepV cleaning models.

This script trains UNet-based models for technical drawing cleaning and artifact removal.
Supports various architectures, mixed precision training, and early stopping.
"""

import argparse
import os
from typing import Any, Optional, Tuple


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Accepts an optional list `args` for programmatic use in tests.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='What model to use, one of [ "UNET,"UNET_MSE,"SmallUnet"]',
        default="UNET",
    )
    parser.add_argument("--n_epochs", type=int, help="Num of epochs for training", default=10)
    parser.add_argument("--datadir", type=str, help="Path to training dataset")
    parser.add_argument("--valdatadir", type=str, help="Path to validation dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--name", type=str, help="Name of the experiment")
    parser.add_argument("--added_part", type=str, default="No", help='["No"],["Yes"]')
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=False,
        help="Enable mixed precision training (FP16/FP32) for improved speed and memory efficiency",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=False,
        help="Enable early stopping based on validation loss",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Number of epochs to wait before early stopping",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-3,
        help="Minimum improvement to reset early stopping patience",
    )

    return parser.parse_args(args)


# TODO: check this metric or change it
# `iou_score` is imported lazily inside `validate` to avoid importing heavy image libs at module import time.


def get_model_and_loss(selected_model_key: str) -> Tuple[Any, Any]:
    """Lazily import models and losses and return the chosen model and loss.

    This avoids importing heavy ML packages at module import time.
    """
    from cleaning.models.SmallUnet.unet import SmallUnet
    from cleaning.models.Unet.unet_model import UNet
    from cleaning.utils.loss import CleaningLoss

    MODEL_LOSS = {
        "UNET": {
            "model": UNet(n_channels=3, n_classes=1, final_tanh=True),
            "loss": CleaningLoss(kind="BCE", with_restore=False),
        },
        "SmallUNET": {
            "model": SmallUnet(),
            "loss": CleaningLoss(
                kind="BCE",
                with_restore=False,
            ),
        },
        "UNET_MSE": {"model": UNet(n_channels=3, n_classes=1), "loss": CleaningLoss(kind="MSE", with_restore=False)},
    }

    if selected_model_key not in MODEL_LOSS:
        available_models = list(MODEL_LOSS.keys())
        raise ValueError(f"Unsupported model type '{selected_model_key}'. Choose from: {available_models}")

    return MODEL_LOSS[selected_model_key]["model"], MODEL_LOSS[selected_model_key]["loss"]


# `parse_args(args=None)` is defined above (accepts an optional args list).


def get_dataloaders(args: argparse.Namespace) -> Tuple[Any, Any]:
    # Local imports to avoid heavy dependencies at module import time
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from cleaning.utils.dataloader import MakeDataSynt

    train_transform = transforms.Compose([transforms.ToTensor()])

    dset_synt = MakeDataSynt(args.datadir, args.datadir, train_transform, 1)
    dset_val_synt = MakeDataSynt(args.valdatadir, args.valdatadir, train_transform)

    #     print(args.batch_size)

    train_loader = DataLoader(
        dset_synt, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False  # 1 for CUDA
    )

    val_loader = DataLoader(
        dset_val_synt, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=False  # 1 for CUDA
    )

    return train_loader, val_loader


def validate(tb: Any, val_loader: Any, model: Any, loss_func: Any, global_step: int) -> float:
    val_loss_epoch = []
    val_iou_extract = []
    # Local imports
    import torch
    import torchvision
    from tqdm import tqdm

    for x_input, y_extract, y_restor in tqdm(val_loader):
        x_input = torch.FloatTensor(x_input).cuda()
        y_extract = y_extract.type(torch.FloatTensor).cuda()

        logits_restor, logits_extract = None, model(x_input)  # restoration + extraction

        # `added_part` is expected to be available on the args passed to training; callers should account for it
        # Lazy import of iou_score to avoid heavy skimage dependency on module import
        from util_files.metrics.raster_metrics import iou_score

        if getattr(val_loader, "added_part", None) == "Yes":
            loss = loss_func(logits_extract, logits_restor, y_extract, y_restor)
            iou_scr = iou_score(
                torch.round(torch.clamp(logits_extract, 0, 1).cpu()).long().numpy(), y_extract.cpu().long().numpy()
            )
        else:
            loss = loss_func(logits_extract, logits_restor, y_restor, y_extract)
            iou_scr = iou_score(
                torch.round(torch.clamp(logits_extract, 0, 1).cpu()).long().numpy(), y_restor.cpu().long().numpy()
            )

        val_iou_extract.append(iou_scr)
        val_loss_epoch.append(loss.cpu().data.numpy())
        del loss

    tb.add_scalar("val_loss", np.mean(val_loss_epoch), global_step=global_step)  # noqa: F821
    tb.add_scalar("val_iou_extract", np.mean(val_iou_extract), global_step=global_step)  # noqa: F821
    out_grid = torchvision.utils.make_grid(logits_extract.unsqueeze(1).cpu())
    input_grid = torchvision.utils.make_grid(x_input.cpu())
    tb.add_image(tag="val_out_extract", img_tensor=out_grid, global_step=global_step)
    tb.add_image(tag="val_input", img_tensor=input_grid, global_step=global_step)

    return np.mean(val_loss_epoch)  # type: ignore # noqa: F821


def save_model(model: Any, path: str) -> None:
    print('Saving model to "%s"' % path)
    import torch

    torch.save(model, path)


def load_model(model: Any, path: str) -> Any:
    print('Loading model from "%s"' % path)
    import torch

    model = torch.load(path)
    return model


def main(args: argparse.Namespace) -> None:
    """Main training function for DeepV cleaning models.

    Trains UNet-based models for technical drawing cleaning and artifact removal.
    Supports various model architectures and training configurations.
    """
    # Local heavy imports
    import torch
    from tensorboardX import SummaryWriter
    from tqdm import tqdm

    from util_files.early_stopping import create_early_stopping_for_cleaning
    from util_files.mixed_precision import MixedPrecisionTrainer

    tb_dir = "/logs/tb_logs_article/cleaning" + (args.name or "")
    tb = SummaryWriter(tb_dir)

    train_loader, val_loader = get_dataloaders(args)

    model, loss_func = get_model_and_loss(args.model)
    model = model.cuda()

    if hasattr(args, "model_path") and args.model_path is not None:
        model = load_model(model, args.model_path)
    model.eval()

    tb.add_text(tag="model", text_string=repr(model))

    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize mixed precision trainer
    mp_trainer = MixedPrecisionTrainer(enabled=getattr(args, "mixed_precision", False))
    print(f"Mixed precision training: {'enabled' if mp_trainer.is_enabled() else 'disabled'}")

    # Initialize early stopping
    early_stopping = (
        create_early_stopping_for_cleaning(
            patience=getattr(args, "early_stopping_patience", 10),
            min_delta=getattr(args, "early_stopping_min_delta", 1e-3),
        )
        if getattr(args, "early_stopping", False)
        else None
    )
    if early_stopping:
        print(f"Early stopping enabled: patience={early_stopping.patience}, min_delta={early_stopping.min_delta}")

    global_step = 0

    for epoch in range(args.n_epochs):
        model.train()
        for x_input, y_extract, y_restor in tqdm(train_loader):
            x_input = torch.FloatTensor(x_input).cuda()
            y_extract = y_extract.type(torch.FloatTensor).cuda()

            with mp_trainer.autocast_context():
                logits_restor, logits_extract = None, model(x_input)  # restoration + extraction

            if args.added_part == "Yes":
                loss = loss_func(logits_extract, logits_restor, y_extract, y_restor)
            else:
                loss = loss_func(logits_extract, logits_restor, y_restor, y_extract)

            mp_trainer.backward(loss)
            mp_trainer.step_optimizer(opt)

            tb.add_scalar("train_loss", loss.cpu().data.numpy(), global_step=global_step)

            if global_step % 500 == 0:
                out_grid = torchvision.utils.make_grid(logits_extract.unsqueeze(1).cpu())  # type: ignore # noqa: F821
                input_grid = torchvision.utils.make_grid(x_input.cpu())  # type: ignore # noqa: F821

                tb.add_image(tag="train_out_extract", img_tensor=out_grid, global_step=global_step)
                tb.add_image(tag="train_input", img_tensor=input_grid, global_step=global_step)

                model.eval()
                with torch.no_grad():
                    val_loss = validate(tb, val_loader, model, loss_func, global_step=global_step)
                model.train()
                save_model(model, os.path.join(tb_dir, "model_it_%s.pth" % global_step))

                # Check early stopping at the end of each epoch
                if early_stopping is not None and (global_step + 1) % (len(train_loader) // args.n_epochs) == 0:
                    epoch = global_step // (len(train_loader) // args.n_epochs)
                    should_stop = early_stopping(val_loss, model, epoch)
                    if should_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        early_stopping.restore_weights(model)
                        break

            del logits_extract
            del logits_restor

            global_step += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
