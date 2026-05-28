import random
import torchvision.transforms.functional as TF
import copy


class RandomRot2DRotation:
    def __call__(self, x):
        k = random.randint(0, 3)  # 0,1,2,3 → 0°,90°,180°,270°
        return TF.rotate(x, angle=90 * k)

class EarlyStopping:
    """Early stopping utility to stop training when validation loss/accuracy stops improving."""
    
    def __init__(self, patience=20, min_delta=0.0, mode='max', restore_best_weights=True, verbose=True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy (higher is better), 'min' for loss (lower is better)
            restore_best_weights: Whether to restore model weights from best epoch
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, score, model, epoch):
        """
        Args:
            score: Current validation score (accuracy or loss)
            model: Model to save weights from
            epoch: Current epoch number
            
        Returns:
            True if early stopping should be triggered, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"✓ Improvement! New best score: {score:.4f} (epoch {epoch})")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\nEarly stopping triggered! Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.early_stop
    
    def _is_better(self, current, best):
        """Check if current score is better than best score."""
        if self.mode == 'max':
            return current > (best + self.min_delta)
        else:  # mode == 'min'
            return current < (best - self.min_delta)
    
    def restore_weights(self, model):
        """Restore model weights to best checkpoint."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"Restored weights from epoch {self.best_epoch} (score: {self.best_score:.4f})")