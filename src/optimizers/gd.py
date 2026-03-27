from .optimizer import Optimizer
import torch


class Gd:
    def __init__(self, eta, optimizer_name="gd", min_alpha=1e-8):
        self.eta = eta
        self.optimizer_name = optimizer_name
        self.min_alpha = min_alpha
        self._adam = None

    def step(self, alpha, dE_dalpha, optimizer_name=None):

        #  Steepest descent update
        if self.optimizer_name == "gd":

            alpha_new = alpha - self.eta * dE_dalpha

            with torch.no_grad():
                alpha_new = torch.clamp(alpha_new, min=self.min_alpha)

            return alpha_new.detach().requires_grad_(True)

        #  Steepest descent with Adam update for learning rate
        elif self.optimizer_name == "adam":

            # create Adam optimizer the first time
            if self._adam is None:
                alpha = alpha.detach().clone().requires_grad_(True)
                self._adam = torch.optim.Adam([alpha], lr=self.eta)

            self._adam.zero_grad()

            alpha.grad = dE_dalpha.detach().clone()

            self._adam.step()

            with torch.no_grad():
                alpha.clamp_(min=self.min_alpha)

            return alpha