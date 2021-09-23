import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_model, match_state_dict_keys


def fgsm(x, y, model, epsilon, crit, mask=None):
    # fix dimensions if the input is not batched
    remove_dim = False
    if len(x.shape) == 3:
        x = x[None, :, :, :]
        remove_dim = True
    if type(y) is int:
        y = torch.tensor([y]).to(x.device)

    # feed input through the network and backprop
    x_ = x.clone().detach().requires_grad_(True)
    y_ = model(x_)
    loss = crit(y_, y)
    loss.backward()

    # apply fgsm
    if mask is not None:
        return_tensor = x + epsilon * torch.mul(mask, torch.sign(x_.grad))
    else:
        return_tensor = x + epsilon * torch.sign(x_.grad)

    return_tensor = torch.clamp(return_tensor, 0, 1)
    # reshape to original form
    if remove_dim:
        return return_tensor[0, :, :, :]
    else:
        return return_tensor


def _get_split_index(n_samples: int, ratio: float):
    return int(max(1, ratio * n_samples))


def _get_input_mask(x: int, ratio: float):
    n_samples = x.shape[0]
    idx = _get_split_index(n_samples, ratio)
    if idx == n_samples:
        return torch.zeros_like(x)
    zeros = torch.zeros_like(x[:idx])
    ones = torch.ones_like(x[idx:])
    return torch.cat((zeros, ones), dim=0)


class AdversarialTransform(nn.Module):

    def __init__(self, epsilon: float, checkpoint_path: str, model_type: str,
                 adv_rate: float = 0.5, mode: str = 'normal'):
        super(AdversarialTransform, self).__init__()
        assert mode in ['normal', '2n', 'null'], '--mode must be either normal, 2n or null'
        assert 0 <= adv_rate <= 1, '--advRatio must be between 0 and 1'
        # load adversarial model
        if epsilon > 0:
            self.adv_model = get_model(model_type)
            state_dict = match_state_dict_keys(torch.load(checkpoint_path)['net'])
            self.adv_model.load_state_dict(state_dict)
        else:
            self.adv_model = None

        self.mode = mode
        self.adv_rate = adv_rate
        self.epsilon = epsilon
        self.__mask = None
        self.crit = nn.CrossEntropyLoss()

    def _mask(self, x):
        if self.adv_rate == 0:
            return None
        if self.__mask is None:
            self.__mask = _get_input_mask(x, 1 - self.adv_rate).to(x.device)
        return self.__mask

    def forward(self, data):
        if self.epsilon == 0:
            return data
        x, y = data
        return fgsm(x, y, self.adv_model, self.epsilon, self.crit, mask=self._mask(x)), y


class AdversarialLabelTransform(nn.Module):

    def __init__(self, mode: str):
        super(AdversarialLabelTransform, self).__init__()
        assert mode in ['normal', '2n', 'null'], '--mode must be either normal, 2n or null'
        self.mode = mode

    def forward(self, x):
        pass