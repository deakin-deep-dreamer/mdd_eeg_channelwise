
import torch
import torch.nn as nn
import torch.nn.functional as F


class DANN_CNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.iter = 0
        self.n_block = args[0]
        self.output_dim = args[1]
        self.log = print if kwargs.get('log') is None else kwargs.get('log')
        conv_layers = []
        in_chan = 1
        out_chan = 8
        for _ in range(1, args[0]+1):            
            conv_layers += [
                nn.Conv1d(
                    in_channels=in_chan, out_channels=out_chan, kernel_size=5, padding="same", bias=False),
                nn.BatchNorm1d(out_chan),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            in_chan = out_chan
            out_chan = in_chan * 2
        out_chan //= 2
        self.encoder = nn.Sequential(*conv_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.decoder = nn.Sequential(
            nn.Linear(out_chan, out_chan//2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_chan//2, self.output_dim),
            nn.Softmax(dim=1),
        )
        self.debug(f"n_block:{args[0]}, output_dim:{args[1]},")

    def forward(self, x):
        self.debug(f"input: {x.shape}")

        out = self.encoder(x)
        self.debug(f"encoder out: {out.shape}")

        out = self.gap(out)
        self.debug(f"GAP out: {out.shape}")

        out = out.view(out.size(0), -1)
        self.debug(f"out flat: {out.shape}")
        
        out = self.decoder(out)
        self.debug(f"src_out: {out.shape}")

        self.iter += 1
        return out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


def test_model():
    net = DANN_CNN(5, 2, log=print)
    print(net)
    x = torch.randn(32, 1, 100*30)
    out = net(x)
    print(f"out {out[0].shape}, {out[1].shape}")


if __name__ == "__main__":
    test_model()