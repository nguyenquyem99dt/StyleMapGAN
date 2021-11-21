import argparse
import torch
from torch import nn
from torch.quantization import quantize_dynamic
from training.model import Generator, Encoder

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.g_ema = Generator(
            train_args.size,
            train_args.mapping_layer_num,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            lr_mul=train_args.lr_mul,
            channel_multiplier=train_args.channel_multiplier,
            normalize_mode=train_args.normalize_mode,
            small_generator=train_args.small_generator,
        )
        self.e_ema = Encoder(
            train_args.size,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            channel_multiplier=train_args.channel_multiplier,
        )
        self.device = device

    def forward(self, original_image, references, masks, shift_values):

        combined = torch.cat([original_image, references], dim=0)

        ws = self.e_ema(combined)
        original_stylemap, reference_stylemaps = torch.split(
            ws, [1, len(ws) - 1], dim=0
        )

        mixed = self.g_ema(
            [original_stylemap, reference_stylemaps],
            input_is_stylecode=True,
            mix_space="demo",
            mask=[masks, shift_values, args.interpolation_step],
        )[0]

        return mixed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, default='100000.pt')
    args = parser.parse_args()

    device = "cpu"
    ckpt = torch.load(args.input)

    train_args = ckpt["train_args"]
    # print("train_args: ", train_args)

    model = Model().to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model.eval()

    output = args.input[:-3] + '_quantized.pt'

    g_ema = quantize_dynamic(model.g_ema, dtype=torch.qint8)
    e_ema = quantize_dynamic(model.e_ema, dtype=torch.qint8)
    torch.save({"g_ema": g_ema.state_dict(),
                "e_ema": e_ema.state_dict(),
                "train_args": train_args},
                output)

    print(f'Successfully! Save quatized model at {output}.')