from abc import ABCMeta, abstractmethod

def lr_poly_exp(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def lr_poly_epoch(base_lr, iter, max_iter, power):
    return base_lr/2.0


def adjust_learning_rate(optimizer, i_iter, args):
    """Sets the learning rate
    Args:
        optimizer: The optimizer
        i_iter: The number of interations
    """
    if args.dataset == "SUNRGBD":
        lr = max(lr_poly_exp(args.learning_rate, i_iter, args.num_steps, args.power), args.min_learining_rate)
    elif args.dataset == "NYUD":
        lr = max(lr_poly_exp(args.learning_rate, i_iter, args.num_steps, args.power), args.min_learining_rate)
    else:
        lr = lr_poly_exp(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def adjust_learning_rate_warmup(optimizer, i_iter, args):
    """Sets the learning rate
    Args:
        optimizer: The optimizer
        i_iter: The number of interations
    """
    args.warmup_steps = 6000
    if i_iter < args.warmup_steps:
        lr = args.learning_rate * (i_iter / args.warmup_steps)
    else:
        lr = max(lr_poly_exp(args.learning_rate, i_iter, args.num_steps, args.power), args.min_learining_rate)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_momentum(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('InPlaceABN') != -1:
        m.momentum = 0.0003