#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
import os
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn, Tensor
from image2latex.model import Image2LatexModel
from data.dataset import LatexDataset, LatexPredictDataset
from data.datamodule import DataModule
from image2latex.text import Text100k, Text170k
import pytorch_lightning as pl
import numpy as np
import wandb  # Ensure wandb is imported if used


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Training and Prediction for Image2LaTeX")

    # Common arguments
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--accumulate-batch", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--data-path", type=str, help="Path to the dataset", default=None)
    parser.add_argument("--img-path", type=str, help="Path to the image folder", default=None)
    parser.add_argument(
        "--predict-img-path", type=str, help="Path to the image(s) for prediction (comma-separated)", default=None
    )
    parser.add_argument(
        "--dataset", type=str, help="Choose dataset [100k, 170k]", default="100k"
    )
    parser.add_argument("--train", action="store_true", help="Flag to enable training")
    parser.add_argument("--val", action="store_true", help="Flag to enable validation")
    parser.add_argument("--test", action="store_true", help="Flag to enable testing")
    parser.add_argument("--predict", action="store_true", help="Flag to enable prediction")
    parser.add_argument("--log-text", action="store_true", help="Flag to log text outputs")
    parser.add_argument("--train-sample", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--val-sample", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--test-sample", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--max-epochs", type=int, default=15, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--model-name", type=str, default="conv_lstm", help="Name of the model")
    parser.add_argument("--enc-type", type=str, default="conv_row_encoder",
                        help="Encoder type [conv_row_encoder, conv_encoder, conv_bn_encoder]")
    parser.add_argument("--enc-dim", type=int, default=512, help="Encoder dimension")
    parser.add_argument("--emb-dim", type=int, default=80, help="Embedding dimension")
    parser.add_argument("--attn-dim", type=int, default=512, help="Attention dimension")
    parser.add_argument("--dec-dim", type=int, default=512, help="Decoder dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--decode-type",
        type=str,
        default="greedy",
        help="Decoding type [greedy, beamsearch]",
    )
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width for beam search")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of layers in the model")
    parser.add_argument("--grad-clip", type=int, default=0, help="Gradient clipping value")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--log-step", type=int, default=100, help="Logging step interval")
    parser.add_argument("--random-state", type=int, default=12, help="Random seed")
    parser.add_argument("--ckpt-path", type=str, help="Path to the model checkpoint", default=None)

    args = parser.parse_args()

    return args, parser


def set_defaults_for_prediction(args, parser):
    """
    Set default values for arguments when prediction mode is enabled.
    """
    if args.predict:
        # Ensure necessary arguments for prediction are provided
        if not args.ckpt_path:
            parser.error("--predict requires --ckpt-path.")
        if not args.predict_img_path:
            parser.error("--predict requires --predict-img-path.")

        # Set defaults specific to prediction
        args.batch_size = 1
        args.accumulate_batch = 1
        args.train_sample = 0
        args.val_sample = 0
        args.test_sample = 0
        args.num_workers = 2
        args.max_epochs = 1
        args.lr = 0.0
        args.img_path = args.img_path or "/kaggle/input/im2latex100k/formula_images_processed/formula_images_processed"
        args.data_path = args.data_path or "/kaggle/input/im2latex-sorted-by-size"
        args.log_step = 1
        args.log_text = False
        args.decode_type = "greedy"  # or set to your preferred default


def main():
    # Parse arguments
    args, parser = parse_arguments()

    # Set defaults if prediction mode is enabled
    set_defaults_for_prediction(args, parser)

    # Set seeds for reproducibility
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    # Initialize text processor based on dataset
    if args.dataset == "100k":
        text = Text100k()
    elif args.dataset == "170k":
        text = Text170k()
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset}")

    # Initialize datasets
    train_set = LatexDataset(
        data_path=args.data_path,
        img_path=args.img_path,
        data_type="train",
        n_sample=args.train_sample,
        dataset=args.dataset,
    )
    val_set = LatexDataset(
        data_path=args.data_path,
        img_path=args.img_path,
        data_type="validate",
        n_sample=args.val_sample,
        dataset=args.dataset,
    )
    test_set = LatexDataset(
        data_path=args.data_path,
        img_path=args.img_path,
        data_type="test",
        n_sample=args.test_sample,
        dataset=args.dataset,
    )
    
    # Handle multiple image paths for prediction
    if args.predict_img_path:
        predict_img_paths = [path.strip() for path in args.predict_img_path.split(',')]
    else:
        predict_img_paths = []
        
    predict_set = LatexPredictDataset(predict_img_paths=predict_img_paths)

    # Initialize DataModule
    dm = DataModule(
        train_set,
        val_set,
        test_set,
        predict_set,
        args.num_workers,
        args.batch_size,
        text,
    )

    # Calculate total steps for learning rate scheduling
    steps_per_epoch = round(len(train_set) / args.batch_size) if args.batch_size > 0 else 0
    total_steps = steps_per_epoch * args.max_epochs

    # Initialize the model
    model = Image2LatexModel(
        lr=args.lr,
        total_steps=total_steps,
        n_class=text.n_class,
        enc_dim=args.enc_dim,
        enc_type=args.enc_type,
        emb_dim=args.emb_dim,
        dec_dim=args.dec_dim,
        attn_dim=args.attn_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sos_id=text.sos_id,
        eos_id=text.eos_id,
        decode_type=args.decode_type,
        text=text,
        beam_width=args.beam_width,
        log_step=args.log_step,
        log_text=args.log_text,
    )

    # Initialize logger
    wandb_logger = pl.loggers.WandbLogger(
        project="image2latex", name=args.model_name, log_model="all"
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # Calculate gradient accumulation steps
    if args.accumulate_batch > 0 and args.batch_size > 0:
        accumulate_grad_batches = args.accumulate_batch // args.batch_size
    else:
        accumulate_grad_batches = 1

    # Initialize Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[lr_monitor],
        max_epochs=args.max_epochs,
        accelerator="auto",
        strategy="dp",
        log_every_n_steps=1,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=accumulate_grad_batches,
        devices=-1,  # Use all available GPUs
    )

    # Load checkpoint if provided
    if args.ckpt_path:
        model = Image2LatexModel.load_from_checkpoint(args.ckpt_path)

    # Perform actions based on flags
    if args.train:
        print("=" * 10 + " [Train] " + "=" * 10)
        trainer.fit(datamodule=dm, model=model, ckpt_path=args.ckpt_path)

    if args.val:
        print("=" * 10 + " [Validate] " + "=" * 10)
        trainer.validate(datamodule=dm, model=model, ckpt_path=args.ckpt_path)

    if args.test:
        print("=" * 10 + " [Test] " + "=" * 10)
        trainer.test(datamodule=dm, model=model, ckpt_path=args.ckpt_path)

    if args.predict:
        print("=" * 10 + " [Predict] " + "=" * 10)
        predictions = trainer.predict(datamodule=dm, model=model, ckpt_path=args.ckpt_path)
        
        # Ensure that predict_img_paths are available
        if not predict_img_paths:
            print("No images provided for prediction.")
            return
        
        # Iterate over image paths and predictions
        for img_path, prediction in zip(predict_img_paths, predictions):
            # Extract base name without extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Define the output .txt file path
            txt_path = os.path.join(os.path.dirname(img_path), f"{base_name}.txt")
            try:
                with open(txt_path, 'w') as f:
                    f.write(prediction)
                print(f"Prediction saved to {txt_path}")
            except Exception as e:
                print(f"Failed to write prediction for {img_path}: {e}")


if __name__ == "__main__":
    main()
