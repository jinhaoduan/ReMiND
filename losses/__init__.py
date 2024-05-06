import torch.optim as optim


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))

def dynamic_lr(optimizer, step, warmup, max_lr, lr_decay_start, decay_step):
    # if step >= warmup and step < lr_decay_start:
    #     return max_lr
    if step < warmup:
        lr = max_lr * min(float(step) / max(warmup, 1), 1.0)
    elif step >= warmup and step < lr_decay_start:
        lr = max_lr
    elif step >= lr_decay_start:
        lr = max_lr * max((1 - ((step - lr_decay_start) / decay_step)), 0)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def warmup_lr(optimizer, step, warmup, max_lr):
    if step > warmup:
        return max_lr
    lr = max_lr * min(float(step) / max(warmup, 1), 1.0)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

