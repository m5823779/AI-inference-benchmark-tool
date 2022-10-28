import os
import sys
import time
import argparse

import torch
import numpy as np
from copy import deepcopy


def model_info(model, verbose=True, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device)  # input
        flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # stride GFLOPS
        img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  # expand if int/float
        fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ''

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")


def main():
    total = 0
    img_size = opt.size if isinstance(opt.size, list) else [opt.size, opt.size]
    model_path = opt.models
    num_infers = opt.num_infers
    device = opt.device

    # select device
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
    cuda = not cpu and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if opt.models:
        model = torch.load(model_path)
    else:
        bar = getattr(sys.modules[__name__], opt.model_name)
        model = bar(opt.weights)

    model_info(model, True, img_size)
    model.to(device)

    dummy_input = torch.rand(1, 3, *img_size).to(device)

    # Warming up
    model(dummy_input)

    for i in range(num_infers):
        start = time.perf_counter()

        model(dummy_input)

        end = (time.perf_counter() - start) * 1000
        total += end
        print('\rInference latency: %.2f ms' % end, end='')
    total /= num_infers
    print(f"\nAverage latency: {total:.2f}ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, help='trace pytorch (.pt) model')

    parser.add_argument('--module_name', type=str, help='import class (module) name')
    parser.add_argument('--import_module', type=str, help='import module path')
    parser.add_argument('--weights', type=str, help='pytorch (.pt) model [only weights]')

    parser.add_argument('--size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--num_infers', default=1000, type=int, help='number of inferences')
    opt = parser.parse_args()
    print(opt)

    assert (not opt.models or not opt.weights), "[--models] [--weights] choose one to load"
    if opt.weights:
        assert (opt.import_module and opt.model_name), "Please enter [--model_name] [--import_module]"
        import_module = "from " + opt.import_module + " import " + opt.model_name
        exec(import_module)

    main()

