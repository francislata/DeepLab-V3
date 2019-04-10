from torch.optim.lr_scheduler import _LRScheduler

class PolynomialLRScheduler(_LRScheduler):
    """This subclass implements the polynomial learning rate scheduler"""
    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        
        super(PolynomialLRScheduler, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        return [self._calculate_polynomial_lr(base_lr) for base_lr in self.base_lrs]

    def _calculate_polynomial_lr(self, lr):
        """Calculates the polynomial learning rate given the current learning rate"""
        return lr * (1.0 - self.last_epoch // self.max_epochs) ** self.power
