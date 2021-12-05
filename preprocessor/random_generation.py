import argparse
import os
import torch
from torch import nn
from torchvision import utils
from tqdm import tqdm
from training.model import Generator, Encoder

class Model(nn.Module):
    def __init__(self, device="cuda"):
        super(Model, self).__init__()
        self.g_ema = Generator(
            args.size,
            args.mapping_layer_num,
            args.latent_channel_size,
            args.latent_spatial_size,
            lr_mul=args.lr_mul,
            channel_multiplier=args.channel_multiplier,
            normalize_mode=args.normalize_mode,
            small_generator=args.small_generator,
        )
        # self.e_ema = Encoder(
        #     args.size,
        #     args.latent_channel_size,
        #     args.latent_spatial_size,
        #     channel_multiplier=args.channel_multiplier,
        # )

    def forward(self, input, mode):
        if mode == "calculate_mean_stylemap":
            truncation_mean_latent = self.g_ema(input, calculate_mean_stylemap=True)

            return truncation_mean_latent

        elif mode == "random_generation":
            z, truncation, truncation_mean_latent = input

            fake_img, _ = self.g_ema(
                z,
                truncation=truncation,
                truncation_mean_latent=truncation_mean_latent,
            )

            return fake_img

def make_noise(batch, latent_channel_size, device):
    return torch.randn(batch, latent_channel_size, device=device)

def save_image(img, path, normalize=True, range=(-1, 1)):
    utils.save_image(
        img,
        path,
        normalize=normalize,
        range=range,
    )

if __name__ == '__main__':
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000)
    # parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--save_image_dir', type=str, default='expr/raw_random_generation')
    parser.add_argument('--ckpt', metavar='CHECKPOINT', required=True)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.save_image_dir, exist_ok=True)

    ckpt = torch.load(args.ckpt)
    train_args = ckpt["train_args"]

    for key in vars(train_args):
        if not (key in vars(args)):
            setattr(args, key, getattr(train_args, key))
    print(args)

    dataset_name = args.dataset
    batch = args.batch

    model = Model().to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    # model.e_ema.load_state_dict(ckpt["e_ema"])
    model.eval()

    with torch.no_grad():
        truncation = 0.7
        truncation_sample = args.num_samples
        truncation_mean_latent = torch.Tensor().to(device)
        for _ in range(truncation_sample // batch):
            z = make_noise(batch, args.latent_channel_size, device)
            partial_mean_latent = model(z, mode="calculate_mean_stylemap")
            truncation_mean_latent = torch.cat(
                [truncation_mean_latent, partial_mean_latent], dim=0
            )
        truncation_mean_latent = truncation_mean_latent.mean(0, keepdim=True)

        torch.manual_seed(args.seed)

        total_images_len = args.num_samples
        num_iter = total_images_len // batch if total_images_len % batch == 0 else total_images_len // batch + 1

        for i in tqdm(range(num_iter)):
            num = batch if total_images_len > batch else total_images_len
            z = make_noise(num, args.latent_channel_size, device)
            total_images_len -= batch

            images = model(
                (z, truncation, truncation_mean_latent),
                mode="random_generation",
            )
            for k, image in enumerate(images):
                save_image(image, f'{args.save_image_dir}/random_generated_{dataset_name}_{i}_{k}.png')

