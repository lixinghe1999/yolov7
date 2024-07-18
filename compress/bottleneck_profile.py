import sys
sys.path.append('./')
from models.yolo import Model
import argparse
from utils.general import check_file, set_logging
from utils.torch_utils import select_device
import torch
import matplotlib.pyplot as plt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolor-csp-c.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()
    
    if opt.profile:
        img = torch.rand(1, 3, 640, 640).to(device)
        y = model(img, profile=True)

    float_rgb = 640*640*3*4 / 1024 / 1024
    bit24_rgb = 640*640*3 / 1024 / 1024

    plt.subplot(121)
    plt.plot(model.tensor_size, label='tensor_size_layer')
    plt.plot(model.new_tensor_size, label='tensor_size_transmit')
    plt.axhline(y=float_rgb, color='r', linestyle='--', label='float_RGB')
    plt.axhline(y=bit24_rgb, color='g', linestyle='--', label='bit24')
    plt.xlabel('layer')
    plt.ylabel('size(MB)')
    plt.ylim(0, max(model.tensor_size) + 1)
    plt.legend()

    plt.subplot(122)
    plt.plot(model.num_related_layers, 'r', label='num_related_layers')
    plt.legend()

    plt.savefig('compress/bottleneck_profile.png')
