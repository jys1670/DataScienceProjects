import torch

"""
Архитектура описана в:
https://arxiv.org/pdf/1611.07004.pdf

BatchNorm2d заменен на InstanceNorm2d, меньше высокочастотных артифактов
"""


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            torch.nn.InstanceNorm2d(out_channels, affine=True),
            torch.nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.transform(x)


class GetDiscriminator(torch.nn.Module):
    def __init__(self, in_channels=3, fmaps=[64, 128, 256, 512]):
        super(GetDiscriminator, self).__init__()
        blocks = []
        self.init_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=2 * in_channels,
                out_channels=fmaps[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            torch.nn.LeakyReLU(0.2),
        )

        for ind, maps_amount in enumerate(fmaps[1:]):
            blocks += [
                ConvBlock(
                    in_channels=fmaps[ind],
                    out_channels=maps_amount,
                    stride=2 if maps_amount != fmaps[-1] else 1,
                )
            ]

        blocks += [
            torch.nn.Conv2d(
                in_channels=fmaps[-1],
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        ]
        self.remaining_blocks = torch.nn.Sequential(*blocks)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.init_block(x)
        x = self.remaining_blocks(x)
        return x


if __name__ == "__main__":
    x = torch.randn([1, 3, 512, 512])
    model = GetDiscriminator()
    print("MODEL: ")
    print(model)
    print("INPUT: ", x.size())
    print("OUTPUT: ", model(x, x).shape)
