import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d

import math


#################
# Discriminator #
#################

class Discriminator(nn.Module):
    """Discriminator Network for Super Resolution"""
    def __init__(self, in_channels, ndf, linear_dim, out_dim, disc_type):
        super(Discriminator, self).__init__()

        self.disc_type = disc_type

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fcn = nn.Sequential(
            nn.Linear(ndf * 8, linear_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(linear_dim, out_dim),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(ndf * 8, out_dim, kernel_size=1, stride=1, padding=0),
        )

        self.patch = nn.Sequential(
            nn.Conv2d(ndf * 8, out_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.main(x)

        if self.disc_type == 'fcn':
            out = adaptive_avg_pool2d(input=out, output_size=1)
            out = torch.flatten(out, 1)
            out = self.fcn(out)

        elif self.disc_type == 'conv':
            out = adaptive_avg_pool2d(input=out, output_size=1)
            out = self.conv(out)

        elif self.disc_type == 'patch':
            out = self.patch(out)

        return out

