#!/usr/bin/env python3
"""
DeepV Data Loading Utilities

High-level data loading utilities for training and evaluation.
Combines multiple datasets, handles memory constraints, and provides
optimized data loading with prefetching and chunking.

Features:
- Combined dataset loading from multiple sources
- Memory-aware batching and chunking
- GPU prefetching for performance
- Train/validation split handling
- Support for synthetic and real datasets

Used by training scripts to efficiently load and preprocess data.
"""

import os
from glob import glob

from torch.utils.data import ConcatDataset, DataLoader

from util_files.data.chunked import ChunkedConcatDatasetLoader, ChunkedDatasetLoader
from util_files.data.prefetcher import CudaPrefetcher
from util_files.data.preprocessed import PreprocessedDataset


def make_combined_loaders(
    data_root,
    train_batch_size,
    val_batch_size,
    memory_constraint,
    shuffle_train=True,
    prefetch=False,
    device=None,
    handcrafted_val_part=0.1,
    mini_val_batches_n_per_subset=None,
):
    # prepare datasets
    _handcrafted_datasets = [
        PreprocessedDataset(data_dir)
        for data_dir in glob(os.path.join(data_root, "preprocessed/synthetic_handcrafted/*"))
    ]
    handcrafted_train = []
    handcrafted_val = []
    for dataset in _handcrafted_datasets:
        handcrafted_train.append(dataset.subset((0, 1 - handcrafted_val_part)))
        handcrafted_val.append(dataset.subset((1 - handcrafted_val_part, 1)))

    datasets_train = handcrafted_train
    dataset_val = ConcatDataset(handcrafted_val)

    # prepare loaders
    train_loader = ChunkedConcatDatasetLoader(
        datasets_train,
        batch_size=train_batch_size,
        memory_constraint=memory_constraint,
        shuffle=shuffle_train,
    )
    val_loader = DataLoader(dataset_val, batch_size=val_batch_size)

    # prepare prefetcher
    if prefetch:
        train_loader = CudaPrefetcher(train_loader, device)
        val_loader = CudaPrefetcher(val_loader, device)

    # prepare mini validation set
    if mini_val_batches_n_per_subset is not None:
        mini_valset_size = val_batch_size * mini_val_batches_n_per_subset
        dataset_val_mini = ConcatDataset([dataset.slice(0, mini_valset_size) for dataset in handcrafted_val])

        val_mini_loader = DataLoader(dataset_val_mini, batch_size=val_batch_size)
        if prefetch:
            val_mini_loader = CudaPrefetcher(val_mini_loader, device)
        return train_loader, val_loader, val_mini_loader
    else:
        return train_loader, val_loader


def make_handcrafted_loaders(
    data_root,
    train_batch_size,
    val_batch_size,
    memory_constraint,
    shuffle_train=True,
    prefetch=False,
    device=None,
    handcrafted_val_part=0.1,
    mini_val_batches_n_per_subset=None,
    handcrafted_train_paths=None,
    handcrafted_val_paths=None,
):
    assert not (None is handcrafted_val_part and None is handcrafted_train_paths and None is handcrafted_val_paths)

    # prepare datasets
    if None is not handcrafted_train_paths and None is not handcrafted_val_paths:
        # ignore handcrafted_val_part and use separate training and validation datasets
        handcrafted_train = [
            PreprocessedDataset(os.path.join(data_root, "preprocessed/synthetic_handcrafted", path))
            for path in handcrafted_train_paths
        ]
        handcrafted_val = [
            PreprocessedDataset(os.path.join(data_root, "preprocessed/synthetic_handcrafted", path))
            for path in handcrafted_val_paths
        ]
    else:
        assert None is not handcrafted_val_part
        if None is handcrafted_train_paths:
            _handcrafted_datasets = [
                PreprocessedDataset(data_dir)
                for data_dir in glob(os.path.join(data_root, "preprocessed/synthetic_handcrafted/*"))
            ]
        else:
            _handcrafted_datasets = [
                PreprocessedDataset(os.path.join(data_root, "preprocessed/synthetic_handcrafted", path))
                for path in handcrafted_train_paths
            ]

        handcrafted_train = []
        handcrafted_val = []
        for dataset in _handcrafted_datasets:
            handcrafted_train.append(dataset.subset((0, 1 - handcrafted_val_part)))
            handcrafted_val.append(dataset.subset((1 - handcrafted_val_part, 1)))

    datasets_train = handcrafted_train
    dataset_val = ConcatDataset(handcrafted_val)

    # prepare loaders
    train_loader = ChunkedConcatDatasetLoader(
        datasets_train,
        batch_size=train_batch_size,
        memory_constraint=memory_constraint,
        shuffle=shuffle_train,
    )
    val_loader = DataLoader(dataset_val, batch_size=val_batch_size)

    # prepare prefetcher
    if prefetch:
        train_loader = CudaPrefetcher(train_loader, device)
        val_loader = CudaPrefetcher(val_loader, device)

    # prepare mini validation set
    if mini_val_batches_n_per_subset is not None:
        mini_valset_size = val_batch_size * mini_val_batches_n_per_subset
        dataset_val_mini = ConcatDataset([dataset.slice(0, mini_valset_size) for dataset in handcrafted_val])
        val_mini_loader = DataLoader(dataset_val_mini, batch_size=val_batch_size)
        if prefetch:
            val_mini_loader = CudaPrefetcher(val_mini_loader, device)
        return train_loader, val_loader, val_mini_loader
    else:
        return train_loader, val_loader


def make_bezier_loaders(
    data_root,
    train_batch_size,
    val_batch_size,
    memory_constraint,
    shuffle_train=True,
    prefetch=False,
    device=None,
    mini_val_batches_n_per_subset=None,
):
    # prepare datasets
    bezier_train = PreprocessedDataset(os.path.join(data_root, "quadratic_bezier_only/train"))
    bezier_val = PreprocessedDataset(os.path.join(data_root, "quadratic_bezier_only/val"))

    #     bezier = PreprocessedDataset(
    #         os.path.join(data_root, 'precision-floorplan.beziers.mini')
    #     )
    #     bezier = PreprocessedDataset(
    #         os.path.join(data_root, 'precision-floorplan.beziers.mini')
    #     )

    # prepare loaders
    loader_kwargs = {
        "batch_size": train_batch_size,
        "memory_constraint": memory_constraint,
        "shuffle": shuffle_train,
    }
    train_loader = ChunkedDatasetLoader(bezier_train, **loader_kwargs)
    val_loader = DataLoader(bezier_val, batch_size=val_batch_size)

    # prepare prefetcher
    if prefetch:
        train_loader = CudaPrefetcher(train_loader, device)
        val_loader = CudaPrefetcher(val_loader, device)

    # prepare mini validation set
    if mini_val_batches_n_per_subset is not None:
        mini_valset_size = val_batch_size * mini_val_batches_n_per_subset
        dataset_val_mini = bezier_val.slice(0, mini_valset_size)

        val_mini_loader = DataLoader(dataset_val_mini, batch_size=val_batch_size)
        if prefetch:
            val_mini_loader = CudaPrefetcher(val_mini_loader, device)
        return train_loader, val_loader, val_mini_loader
    else:
        return train_loader, val_loader


prepare_loaders = {
    "combined": make_combined_loaders,
    "handcrafted": make_handcrafted_loaders,
    "bezier": make_bezier_loaders,
}
