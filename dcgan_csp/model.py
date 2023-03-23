import torch
import torch.nn as nn


class CSPup(nn.Module):
    def __init__(self, conv_dims):
        super(CSPup, self).__init__()
        self.conv_dims = conv_dims

        self.left_1_transpose = nn.ConvTranspose2d(conv_dims // 2, conv_dims // 2, 4, 2, 1, bias=False)

        self.right = nn.Sequential(
            nn.Conv2d(conv_dims // 2, conv_dims // 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(conv_dims // 2, conv_dims // 2, 4, 2, 1, bias=False),
            nn.Conv2d(conv_dims // 2, conv_dims // 2, 3, padding='same'),
            nn.ReLU(True),
            nn.Conv2d(conv_dims // 2, conv_dims // 2, 3, padding='same')
        )

    def forward(self, z):
        left = z[:, :self.conv_dims // 2, :, :]
        left = self.left_1_transpose(left)

        right = z[:, self.conv_dims // 2:, :, :]

        right = self.right(right)

        z = torch.add(left, right)

        return z


# class CSPGenerator(nn.Module):
#     def __init__(self):
#         super(CSPGenerator, self).__init__()
#
#         self.layer_1_transpose = nn.ConvTranspose2d(VECTOR_SIZE, GEN_FEATURES * 8, 4, 1, 0, bias=False)
#         self.layer_2_batchnorm = nn.BatchNorm2d(GEN_FEATURES * 8)
#         self.layer_3_csp_up = CSPup(GEN_FEATURES * 8)
#         self.layer_4_csp_up = CSPup(GEN_FEATURES * 4)
#         self.layer_5_csp_up = CSPup(GEN_FEATURES * 2)
#         self.layer_6_csp_up = CSPup(GEN_FEATURES)
#         self.layer_7_conv = nn.Conv2d(GEN_FEATURES // 2, CHANNELS, 3, 1, 1, bias=False)
#
#     def forward(self, array):
#         array = self.layer_1_transpose(array)
#         array = functional.relu(self.layer_2_batchnorm(array))
#         array = self.layer_3_csp_up(array)
#         array = self.layer_4_csp_up(array)
#         array = self.layer_5_csp_up(array)
#         array = self.layer_6_csp_up(array)
#         array = torch.tanh(self.layer_7_conv(array))
#
#         return array

class Generator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            CSPup(ngf * 8),
            nn.Dropout(p=0.2),
            CSPup(ngf * 4),
            nn.Dropout(p=0.2),
            CSPup(ngf * 2),
            nn.Dropout(p=0.5),
            CSPup(ngf),
            nn.Conv2d(ngf // 2, nc, 3, 1, 1, bias=False),
            nn.Tanh()
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
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
