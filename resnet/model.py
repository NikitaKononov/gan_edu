import torch.nn as nn


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, bias=False, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU(True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Conv2dG(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, bias=False, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU(True)
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv2dD(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, bias=False, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            Conv2dTranspose(nz, ngf * 8, 4, 1, 0),
            nn.Dropout(p=0.1),
            Conv2dG(ngf * 8, ngf * 8, 3, 1, 1, residual=True),
            Conv2dG(ngf * 8, ngf * 8, 3, 1, 1, residual=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.Dropout(p=0.1),
            Conv2dG(ngf * 4, ngf * 4, 3, 1, 1, residual=True),
            Conv2dG(ngf * 4, ngf * 4, 3, 1, 1, residual=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.Dropout(p=0.1),
            Conv2dG(ngf * 2, ngf * 2, 3, 1, 1, residual=True),
            Conv2dG(ngf * 2, ngf * 2, 3, 1, 1, residual=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.Dropout(p=0.1),
            Conv2dG(ngf, ngf, 3, 1, 1, residual=True),
            Conv2dG(ngf, ngf, 3, 1, 1, residual=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            Conv2dD(ndf, ndf * 2, 4, 2, 1),
            nn.Dropout(p=0.1),
            Conv2dD(ndf * 2, ndf * 2, 3, 1, 1, residual=True),
            Conv2dD(ndf * 2, ndf * 2, 3, 1, 1, residual=True),
            # state size. (ndf*2) x 16 x 16
            Conv2dD(ndf * 2, ndf * 4, 4, 2, 1),
            nn.Dropout(p=0.1),
            Conv2dD(ndf * 4, ndf * 4, 3, 1, 1, residual=True),
            Conv2dD(ndf * 4, ndf * 4, 3, 1, 1, residual=True),
            # state size. (ndf*4) x 8 x 8
            Conv2dD(ndf * 4, ndf * 8, 4, 2, 1),
            nn.Dropout(p=0.1),
            Conv2dD(ndf * 8, ndf * 8, 3, 1, 1, residual=True),
            Conv2dD(ndf * 8, ndf * 8, 3, 1, 1, residual=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
