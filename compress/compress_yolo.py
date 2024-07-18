import sys
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

from models.yolo import Model, Detect, IAuxDetect, IBin, IDetect, IKeypoint
import torch.nn as nn
import torchvision
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
import torch
import math
from utils.general import check_file, set_logging
from utils.torch_utils import select_device
import argparse
import os
class Bottleneck(nn.Module):
    def __init__(self, input_N, N):
        super(Bottleneck, self).__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.encode = nn.Sequential(
            nn.Conv2d(input_N, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, input_N, kernel_size=5, padding=2, output_padding=1, stride=2)
        )
    def forward(self, x):
       y = self.encode(x)
       y_hat, y_likelihoods = self.entropy_bottleneck(y)
       x_hat = self.decode(y_hat)
       return x_hat, y_likelihoods
class Compress_YOLO(Model):
    '''
    Inherit the YOLO model
    '''
    def __init__(self, cfg='yolov7.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        self.bottle_neck_idx = 1
        super().__init__(cfg, ch, nc, anchors)
        self.number_layer = len(self.model)
        self.bottleneck_layer = Bottleneck(32, 128)
        self.forward_once = self.forward_once_compress

    def forward_once_split(self, x, start_idx, end_idx, y=None):
        if y is None:
            y, dt = [], []  # outputs
        self.tensor_size = [x.nelement() * x.element_size() / 1024 / 1024]
        for i in range(start_idx, end_idx):  # run
            m = self.model[i]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced=False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect) or isinstance(m, IKeypoint):
                    break

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

            # save all the tensor size
        return x, y
    def forward_once_compress(self, x, profile=False):
        '''
        Single device training and inference
        '''
        encode_x, y = self.forward_once_split(x, 0, self.bottle_neck_idx)

        x_hat = self.bottleneck_layer.encode(encode_x)
        x_hat, x_likelihoods = self.bottleneck_layer.entropy_bottleneck(x_hat)
        x_hat = self.bottleneck_layer.decode(x_hat)
        
        x, _ = self.forward_once_split(x_hat, self.bottle_neck_idx, self.number_layer, y)
        if self.training:
            return x, x_likelihoods, encode_x, x_hat
        else:
            return x
    def compress_loss(self, x_likelihoods, encode_x, x_hat):
        '''
        Don't forget auxillary loss: aux_loss = net.entropy_bottleneck.loss()
        '''
        N, _, H, W = encode_x.size()
        num_pixels = N * H * W
        bpp_loss = torch.log(x_likelihoods).sum() / (-math.log(2) * num_pixels)
        mse_loss = nn.functional.mse_loss(x_hat, encode_x)
        return bpp_loss + mse_loss
    def encode(self, x):
        x, y = self.forward_once_split(x, 0, self.bottle_neck_idx)
        encode_x = self.bottleneck_layer.encode(x)
        encode_strings = self.bottleneck_layer.entropy_bottleneck.compress(encode_x)
        return encode_strings, encode_x.size()[2:]
    def decode(self, encode_strings, shape):
        '''
        shape = encode_x.size()[2:]
        '''
        encode_x = self.bottleneck_layer.entropy_bottleneck.decompress(encode_strings, shape)
        x_hat = self.bottleneck_layer.decode(encode_x)

        y = [None for _ in range(self.bottle_neck_idx-1)] + [x_hat] # temporal workaround, only store one layer for later inference
        x, _ = self.forward_once_split(x_hat, self.bottle_neck_idx, self.number_layer, y)
        return x
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolor-csp-c.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    img = torch.rand(1, 3, 640, 640).to(device)

    # Create model
    model = Model(opt.cfg).to(device)

    compress_model = Compress_YOLO(opt.cfg).to(device)
    compress_model.load_state_dict(model.state_dict(), strict=False)

    model.eval()
    compress_model.eval()
    
    # full model
    y = model(img)[0]
    # full model with compress but on the same device
    compress_y = compress_model(img)[0]

    compress_model.bottleneck_layer.entropy_bottleneck.update()
    encode_x, shape = compress_model.encode(img)
    # save encode_x as [string]
    with open('output.bin', 'wb') as file:
        file.write(encode_x[0])
    # save image as file
    torchvision.utils.save_image(img, 'input.jpg')
    # print the bytes size and image size, without training we may increase the size
    print(shape, os.path.getsize('output.bin'), os.path.getsize('input.jpg'))
    os.remove('output.bin')
    os.remove('input.jpg')
    encode_x = [open('output.bin', 'rb').read()]
    decode_y = compress_model.decode(encode_x, shape)[0]

    error = torch.abs(y - compress_y).mean()
    error_decode = torch.abs(y - decode_y).mean()
    print(f'Error: {error}', f'Error_decode: {error_decode}')