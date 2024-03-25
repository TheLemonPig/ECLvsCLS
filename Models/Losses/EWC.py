import torch


class EWC(object):
    """
    Use this class in Main by initializing with relevant parameters

    During training, append the loss calculated from the penalty function
    after the criterion and before the optimization step
    """

    def __init__(self, model, dataloaders, criterion, fisher_multiplier=1000):
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.fisher_multiplier = fisher_multiplier
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._precision_matrices = self._calculate_precision_matrices()

    def _calculate_precision_matrices(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)
        self.model.eval()
        for dataloader in self.dataloaders:
            for inputs, targets in dataloader:
                self.model.zero_grad()
                inputs = inputs.cuda()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                for n, p in self.model.named_parameters():
                    if p.requires_grad:
                        precision_matrices[n] += p.grad.data ** 2 / len(dataloader)
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self):
        loss = 0
        for n, p in self.params.items():
            loss += (self._precision_matrices[n] * (p - p.detach()) ** 2).sum()
        return self.fisher_multiplier * loss
