# Constants
POWER = 0.9

class PolynomialLRScheduler(object):
    def __init__(self, optimizer, learning_rate, num_iters, num_epochs, power=POWER):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.max_iters = self.num_iters * num_epochs
        self.power = power

    def step(self, current_epoch, current_iter):
        iter = current_epoch * self.num_iters + current_iter
        learning_rate = self.learning_rate * pow(1 - float(iter) / self.max_iters, self.power)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate