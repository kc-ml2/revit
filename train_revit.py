import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, OneCycleLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from budget_cli import add_budget_arguments, budget_config, model_config, smooth_targets
from budget_revit import BudgetAdaptiveREViT
from datasets import get_rot_mnist_loaders, get_cifar10_loaders, get_pcam_loaders
from group_space import get_gspace
from revit_gcsa import Rot2DTransformer, count_parameters
from utils import count_flops


device = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    'cifar10': (get_cifar10_loaders, 3, 10, 32),
    'rotmnist': (get_rot_mnist_loaders, 1, 10, 28),
    'pcam': (get_pcam_loaders, 3, 2, 96),
}


def get_loaders(dataset, batch_size=128, num_workers=8, pin_memory=True, data_root=None):
    loader = DATASETS[dataset][0]
    root = {} if data_root is None else {'root': data_root}
    return loader(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, **root)


def held_out_loaders(train_loader, test_loader, fraction, seed):
    train_set = train_loader.dataset
    count = round(len(train_set) * fraction)
    if not 0 < count < len(train_set):
        raise ValueError('The validation fraction must leave nonempty train and validation sets')
    indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(seed)).tolist()
    val_set = copy.copy(train_set)
    val_set.transform = test_loader.dataset.transform
    options = dict(num_workers=train_loader.num_workers, pin_memory=train_loader.pin_memory)
    return (
        DataLoader(Subset(train_set, indices[count:]), batch_size=train_loader.batch_size, shuffle=True, **options),
        DataLoader(Subset(val_set, indices[:count]), batch_size=test_loader.batch_size, **options),
    )


def train_step(model, optimizer, x, y, device="cuda", scaler=None, label_smoothing=0.1,
               adaptive=False, num_classes=10):
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda" or not adaptive):
        if adaptive:
            _, losses = model(x, targets=smooth_targets(y, num_classes, label_smoothing))
            loss = losses.mean()
        else:
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=label_smoothing)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

@torch.no_grad()
def test_step(model, x, y, device="cuda", budget=None):
    x = x.to(device)
    y = y.to(device)

    with autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda" or budget is None):
        out = model(x) if budget is None else model(x, budget=budget)
        loss = F.cross_entropy(out, y)

    preds = out.argmax(dim=1)
    correct = (preds == y).sum().item()
    return loss.item(), correct, y.size(0)


def revit_train(
    dataset='rotmnist',
    channels=64,
    heads=8,
    depth=4,
    num_epochs=200,
    warmup_epochs=5,
    group_str="C4",
    pretrained=False,
    checkpoint_path=None,
    one_cycle=False,
    lr=3e-4,
    patience=20,
    min_delta=0.001,
    saved_epoch=None,
    batch_size=128,
    downsize=1,
    use_conv_attn=True,
    conv_kernel_size=5,
    adaptive_config=None,
    budget=1.0,
    num_workers=8,
    data_root=None,
    output_dir='.',
    validation_fraction=0.0,
    seed=42,
):

    train_loader, test_loader = get_loaders(dataset, batch_size=batch_size, num_workers=num_workers,
                                           pin_memory=device == 'cuda', data_root=data_root)
    if validation_fraction:
        train_loader, test_loader = held_out_loaders(train_loader, test_loader, validation_fraction, seed)
    _, in_channels, num_classes, image_size = DATASETS[dataset]
    if adaptive_config is None:
        model = Rot2DTransformer(
            depth=depth, in_channels=in_channels, channels=channels, heads=heads,
            gspace=get_gspace(group_str), downsize=downsize,
            use_conv_attn=use_conv_attn, conv_kernel_size=conv_kernel_size,
        )
        # Preserve module-prefixed state dictionaries for the legacy model.
        model = torch.nn.DataParallel(model)
    else:
        model = BudgetAdaptiveREViT(adaptive_config)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    model.to(device)
    count_parameters(model)
    count_flops(model, (in_channels, image_size, image_size))

    if pretrained:
        if not checkpoint_path:
            raise ValueError('checkpoint_path is required for pretraining')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        weights = checkpoint.get('model_state_dict', checkpoint)
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        core.load_state_dict({key.removeprefix('module.'): val for key, val in weights.items()})

    output = Path(output_dir)
    for subdir in ('checkpoints', 'es_models'):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    width_tag = str(channels) if adaptive_config is None else '-'.join('x'.join(map(str, p)) for p in adaptive_config.profiles)
    run_name = f'{dataset}_{group_str}_{depth}_{width_tag}_{heads}_{one_cycle}'

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


    if one_cycle:
        scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_loader),
                               pct_start=0.1, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0)
    else:  # LEARNING RATE SCHEDULER WITH WARMUP
        warmup_scheduler = LinearLR(optimizer, start_factor=0.03, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)  # 200 - 5 warmup
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    best_accuracy = -1.0
    writer = SummaryWriter(log_dir=output / 'runs' / run_name)
    scaler = GradScaler(device, enabled=device == 'cuda')
    metadata = model_config(model)

    for epoch in range(num_epochs):
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        model.train()
        total_loss = 0.0
        with tqdm(train_loader, unit='batch', desc=f'Epoch {epoch}') as batches:
            for x, y in batches:
                loss = train_step(model, optimizer, x, y, device=device, scaler=scaler,
                                  adaptive=adaptive_config is not None, num_classes=num_classes)
                if one_cycle:
                    scheduler.step()
                total_loss += loss
                batches.set_postfix(loss=loss)
        if not one_cycle:
            scheduler.step()

        model.eval()
        test_loss = test_correct = test_total = 0
        for x, y in test_loader:
            loss, correct, count = test_step(model, x, y, device=device,
                                             budget=budget if adaptive_config is not None else None)
            test_loss += loss * count
            test_correct += correct
            test_total += count
        accuracy = test_correct / test_total
        validation_loss = test_loss / test_total
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            best_model = copy.deepcopy(model.state_dict())
            torch.save({
                **metadata,
                'epoch': best_epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': validation_loss,
            }, output / 'checkpoints' / f'{run_name}_{best_epoch}.pt')

        train_loss = total_loss / len(train_loader)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('test_loss', validation_loss, epoch)
        writer.add_scalar('test_accuracy', accuracy, epoch)
        writer.add_scalar('best_accuracy', best_accuracy, epoch)
        print(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Eval Loss: {validation_loss:.4f} | Accuracy: {accuracy:.4f}')

    writer.close()
    weights = {**metadata, 'model_state_dict': best_model} if metadata else best_model
    torch.save(weights, output / 'es_models' / f'{run_name}_{best_epoch}.pth')
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train REViT on CIFAR-10, Rotated MNIST, or PCam')
    parser.add_argument('--dataset', choices=DATASETS, default='rotmnist')
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir', default='.')
    parser.add_argument('--group-str', default='C4')
    parser.add_argument('--channels', type=int, default=12, help='Legacy regular fields')
    parser.add_argument('--heads', type=int, nargs='+', default=(6,))
    parser.add_argument('--depths', type=int, nargs='+', default=(4,))
    parser.add_argument('--downsize', type=int, default=1)
    parser.add_argument('--conv-kernel-size', type=int, default=5)
    parser.add_argument('--conv-attn', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--window-size', type=int, help='Use a windowed adaptive model')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--warmup-epochs', type=int, default=30)
    parser.add_argument('--one-cycle', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-fraction', type=float, default=0.0,
                        help='Hold out part of CIFAR-10 or Rotated MNIST training data for model selection')
    parser.add_argument('--pretrained-path')
    add_budget_arguments(parser)
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.budget <= 1:
        parser.error('--epochs must be positive and --budget must lie in [0, 1]')
    if not args.one_cycle and not 0 <= args.warmup_epochs < args.epochs:
        parser.error('--warmup-epochs must be smaller than --epochs with --no-one-cycle')
    if not 0 <= args.validation_fraction < 1 or (args.dataset == 'pcam' and args.validation_fraction):
        parser.error('Use a validation fraction in [0, 1) for CIFAR-10 or Rotated MNIST')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    adaptive_config = None
    if args.budget_profiles is not None:
        _, in_channels, num_classes, image_size = DATASETS[args.dataset]
        adaptive_config = budget_config(
            args, in_channels=in_channels, num_classes=num_classes, image_size=image_size,
            depths=tuple(args.depths), heads=tuple(args.heads), window_size=args.window_size,
            qkv_kernel=args.conv_kernel_size if args.conv_attn else 1, stem_downsample=args.downsize,
        )
    elif len(args.heads) != 1 or len(args.depths) != 1 or args.window_size is not None:
        parser.error('Multiple stages and window attention require --budget-profiles')
    revit_train(
        dataset=args.dataset, channels=args.channels, heads=args.heads[0], depth=args.depths[0],
        num_epochs=args.epochs, warmup_epochs=args.warmup_epochs, one_cycle=args.one_cycle,
        lr=args.lr, batch_size=args.batch_size, group_str=args.group_str, downsize=args.downsize,
        use_conv_attn=args.conv_attn, conv_kernel_size=args.conv_kernel_size,
        pretrained=args.pretrained_path is not None, checkpoint_path=args.pretrained_path,
        adaptive_config=adaptive_config, budget=args.budget, num_workers=args.num_workers,
        data_root=args.data_root, output_dir=args.output_dir,
        validation_fraction=args.validation_fraction, seed=args.seed,
    )


if __name__ == '__main__':
    main()
