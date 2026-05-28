import torch
from torch.nn.functional import leaky_relu
from revit_gcsa import Rot2DTransformer
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, OneCycleLR
import copy
from datasets import get_rot_mnist_loaders, get_cifar10_loaders, get_pcam_loaders
from group_space import get_gspace
from revit_gcsa import count_parameters
from torch.amp import GradScaler
from utils import EarlyStopping
from torch.amp import GradScaler, autocast
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

def get_loaders(dataset, batch_size=128, num_workers=8, pin_memory=True):
    if dataset == 'cifar10':
        return get_cifar10_loaders(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    elif dataset == 'rotmnist':
        return get_rot_mnist_loaders(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    elif dataset == 'pcam':
        return get_pcam_loaders(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

def train_step(model, optimizer, x, y, device="cuda",scaler=None, label_smoothing=0.1):
    x = x.to(device)
    y = y.to(device)

    with autocast(device_type=device, dtype=torch.bfloat16):
        out = model(x)          # GeometricTensor
        loss = F.cross_entropy(out, y, label_smoothing=label_smoothing)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    return loss.item()

@torch.no_grad()
def test_step(model, x, y, device="cuda"):
    x = x.to(device)
    y = y.to(device)

    with autocast(device_type=device, dtype=torch.bfloat16):
        out = model(x)
        loss = F.cross_entropy(out, y)
   
    preds = out.argmax(dim=1)
    correct = (preds == y).sum().item()
    # total += y.size(0)
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
    conv_kernel_size=5
):

    train_loader, test_loader = get_loaders(dataset, batch_size=batch_size)
    gspace = get_gspace(group_str)
    # Multi-GPU model loading
    model = Rot2DTransformer(
        depth=depth,
        in_channels=1 if dataset == 'rotmnist' else 3,
        channels=channels,
        heads=heads,
        gspace=gspace,
        downsize=downsize,
        use_conv_attn=use_conv_attn,
        conv_kernel_size=conv_kernel_size,
    )
    print("GPU's available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)  # Required for multi-GPU
    model.to(device)
    _,_ = count_parameters(model)

    if pretrained:
        if checkpoint_path:
            try:
                # Load the model state dict from the specified checkpoint path
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded successfully from {checkpoint_path}")
            except Exception as e:
                print(f"Error loading model from {checkpoint_path}: {e}. Proceeding without pretraining.")
        else:
            print("pretrained=True but no checkpoint_path provided. Proceeding without pretraining.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)


    if one_cycle:
        scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_loader),
                               pct_start=0.1, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0)
    else:  # LEARNING RATE SCHEDULER WITH WARMUP
        warmup_scheduler = LinearLR(optimizer, start_factor=0.03, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)  # 200 - 5 warmup
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    best_accuracy = 0
    writer = SummaryWriter(log_dir=f"runs/{dataset}_{group_str}_{depth}_{channels}_{heads}_{one_cycle}", comment=f"cifar10_es_transformer_{group_str}_{depth}_{channels}_{heads}_{one_cycle}_{lr}")
    scaler = GradScaler(device)

    for epoch in range(num_epochs):
        writer.add_scalar(f"lr", optimizer.param_groups[0]['lr'], epoch) 
        model.train()
        total_loss = 0
        with tqdm(train_loader, unit='batch') as tepoch:
            for x, y in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                loss = train_step(model, optimizer, x, y, device=device, scaler=scaler, label_smoothing=0.1)
                if one_cycle:
                    scheduler.step()
                total_loss += loss
                tepoch.set_postfix(loss=loss)
                
        if not one_cycle:
            scheduler.step()

        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        if epoch % 1 == 0:              # change to run validation every n-th epoch
          for x, y in test_loader:
              loss, correct, total = test_step(model, x, y)
              test_loss += loss
              test_total += total
              test_correct += correct
          if test_correct / test_total > best_accuracy:
              best_accuracy = test_correct / test_total
              best_model = copy.deepcopy(model.state_dict())
              best_epoch = epoch
              torch.save({
                        'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': test_loss / len(test_loader),
                        },f"checkpoints/{dataset}_{group_str}_{depth}_{channels}_{heads}_{best_epoch}")

          print(f"Epoch {epoch:03d} | Test Loss: {test_loss / len(test_loader):.4f} | Accuracy: {test_correct / test_total:.4f}")
          writer.add_scalar(f"test_loss", test_loss / len(test_loader), epoch)
          writer.add_scalar(f"test_accuracy", test_correct / test_total, epoch)
          writer.add_scalar(f"best_accuracy", best_accuracy, epoch)  


        print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(train_loader):.4f}")

        # print(f"Epoch {epoch:03d} | Loss: {total_loss / len(train_loader):.4f}")
        writer.add_scalar(f"train_loss", total_loss / len(train_loader), epoch) 

    writer.close()
    torch.save(best_model, f"es_models/{dataset}_{group_str}_{depth}_{channels}_{heads}_{best_epoch}.pth")

    return test_correct / test_total    

if __name__ == "__main__":
    
    dataset = 'rotmnist'     # 'cifar10' or 'pcam' or 'rotmnist'
    group_str = "C4"
    depth = 4
    channels = 12
    heads = 6
    num_epochs = 300
    warmup_epochs = 30
    patience = 20
    min_delta = 0.001
    one_cycle = True
    lr = 5e-3
    batch_size = 64
    downsize = 2
    use_conv_attn = True
    conv_kernel_size = 5
    pretrained = False
    saved_epoch = None
    checkpoint_path = None

    results = []

    print(f"\n{'#' * 60}")
    print(
        f"Configuration group={group_str}, depth={depth}, channels={channels}, heads={heads}, "
        f"lr={lr}, one_cycle={one_cycle}, batch_size={batch_size}, downsize={downsize}, "
        f"use_conv_attn={use_conv_attn}, conv_kernel_size={conv_kernel_size}"
    )
    print(f"{'#' * 60}")

    final_acc = revit_train(
        dataset=dataset,
        channels=channels,
        heads=heads,
        depth=depth,
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        group_str=group_str,
        lr=lr,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        saved_epoch=saved_epoch,
        one_cycle=one_cycle,
        patience=patience,
        min_delta=min_delta,
        batch_size=batch_size,
        downsize=downsize,
        use_conv_attn=use_conv_attn,
        conv_kernel_size=conv_kernel_size,
    )
    results.append(
        {
            "depth": depth,
            "channels": channels,
            "heads": heads,
            "final_accuracy": final_acc,
            "patience": patience,
            "min_delta": min_delta,
        }
    )
    print(f"\n✓ Completed: group={group_str}, depth={depth}, channels={channels}, heads={heads}, acc={final_acc:.4f}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("SUMMARY OF RUN: depth={depth}, channels={channels}, heads={heads}")
    print(f"{'=' * 60}")
    for r in results:
        if r["final_accuracy"] is not None:
            print(
                f"group={group_str}, depth={r['depth']:2d}, channels={r['channels']:2d}, heads={r['heads']:2d} -> "
                f"Accuracy: {r['final_accuracy']:.4f}"
            )
        else:
            print(
                f"group={group_str}, depth={r['depth']:2d}, channels={r['channels']:2d}, heads={r['heads']:2d} -> "
                f"FAILED: {r.get('error', 'Unknown error')}"
            )
