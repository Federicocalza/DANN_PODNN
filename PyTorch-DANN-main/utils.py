import math
from torch.autograd import Function
from torch import nan_to_num


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = (grad_output.neg() * ctx.lamda)
        return output, None


def adjust_alpha(i, epoch, min_len, nepochs):
    p = float(i + epoch * min_len) / nepochs / min_len
    o = -10
    alpha = 2. / (1. + math.exp(o * p)) - 1
    return alpha


def adjust_alpha_log_growth(i, epoch, min_len, nepochs, branch_scaling):
    p = float(i + epoch * min_len) / nepochs / min_len  
    alpha = math.log(1 + p) / (branch_scaling*math.log(2))
    return alpha

def replace_nan_inf_hook(module, input, output):
    return nan_to_num(output, nan=0.1, posinf=0.99, neginf=-0.99)