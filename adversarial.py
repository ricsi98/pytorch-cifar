import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_model

class AdversarialTransform(nn.Module):

    def __init__(self, epsilon: float, checkpoint_path: str, model_type: str):
        super(AdversarialTransform, self).__init__()
        # load adversarial model
        if epsilon > 0:
            self.adv_model = get_model(model_type)
            self.adv_model.load_state_dict(torch.load(checkpoint_path))
        else:
            self.adv_model = None

        self.epsilon = epsilon
        self.crit = nn.CrossEntropyLoss()


    def forward(self, data):
        if self.epsilon == 0:
            return data
        x, y = data
        x_ = torch.tensor(x, requires_grad=True)
        y_ = self.adv_model(x_)
        loss = self.crit(y_, y)
        loss.backward()
        return x + self.epsilon * torch.sign(x_.grad), y