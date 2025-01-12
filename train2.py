import contextlib
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import transformers
from copy import deepcopy
from torchvision import transforms

from imgproc import generate_crop_size_list
from dit import NextDiT
from transport import training_losses

# Globals
image_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def main():
    # Setup an experiment folder:

    checkpoint_dir = os.path.join("checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # create tokenizers
    # Load the tokenizers
    tokenizer = transformers.T5Tokenizer.from_pretrained("google-t5/t5-small").to(device)
    tokenizer.padding_side = "right"

    # create text encoders
    # text_encoder

    text_encoder = transformers.T5EncoderModel.from_pretrained("google-t5/t5-small").to(device)

    print(f"text encoder: {type(text_encoder)}")

    model = NextDiT(use_flash_attn=False, qk_norm=True, n_layers=2, cap_feat_dim=512, n_heads=4).to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant
    # learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.999)

    checkpoint_path = "dit_checkpoint.pth"

    # Check if the checkpoint file exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Load the model state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load the optimizer state dict
        opt.load_state_dict(checkpoint['optimizer_state_dict'])

        # Optionally, load the step if you want to resume from that exact point
        start_step = checkpoint['step']

        model = model.to(device)

        print(f"Checkpoint loaded successfully, resuming from step {start_step}")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Starting training from scratch.")

    print(f"DiT Parameters: {model.parameter_count():,}")
    model_patch_size = model.patch_size

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model).to(device)

    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant
    # learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.999)

    # Setup data:
    print("Creating data transform...")
    patch_size = 8 * model_patch_size
    max_num_patches = round((image_size / patch_size) ** 2)
    print(f"Limiting number of patches to {max_num_patches}.")
    crop_size_list = generate_crop_size_list(max_num_patches, patch_size)
    print("List of crop sizes:")
    for i in range(0, len(crop_size_list), 6):
        print(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[i: i + 6]]))

    # image_transform = transforms.Compose(
    #     [
    #         transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list)),
    #         # transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    #     ]
    # )
    #
    # dataset = MyDataset(
    #     args.data_path,
    #     item_processor=T2IItemProcessor(image_transform),
    #     cache_on_disk=args.cache_data_on_disk,
    # )

    # loader = DataLoader(
    #     dataset,
    #     batch_size=local_batch_size,
    #     sampler=sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     collate_fn=dataloader_collate_fn,
    # )

    # Define transforms for CIFAR
    cifar_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),  # Normalize to [-1, 1]
        ]
    )

    # Load the CIFAR-10 dataset
    dataset = CIFAR10(
        root="cifar_data",  # Directory to store CIFAR data
        train=True,  # Use training set
        download=True,  # Download data if not available
        transform=cifar_transform,  # Apply defined transforms
    )

    # Prepare a DataLoader
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    # important! This enables embedding dropout for classifier-free guidance
    model.train()

    for step, (x, caps) in enumerate(loader):
        # caps: List[str]
        x = [img.to(device, non_blocking=True) for img in x]
        label_texts = [dataset.classes[label] for label in caps.tolist()]  # Map integers to class names
        with torch.no_grad():
            vae_scale = 0.13025
            # Map input images to latent space + normalize latents:
            x = [vae.encode(img[None]).latent_dist.sample().mul_(vae_scale)[0] for img in x]

        with torch.no_grad():
            cap_feats, cap_mask = encode_prompt(label_texts, text_encoder, tokenizer, 0.001)

        loss_item = 0.0
        opt.zero_grad()

        model_kwargs = dict(cap_feats=cap_feats, cap_mask=cap_mask)
        with {
            "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
            "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
            "fp32": contextlib.nullcontext(),
            "tf32": contextlib.nullcontext(),
        }["fp16"]:
            loss_dict = training_losses(model, x, model_kwargs)
        loss = loss_dict["loss"].sum()
        print(f"Loss {loss}")
        loss_item += loss.item()
        loss.backward()

        # grad_norm = calculate_l2_grad_norm(model)
        # if grad_norm > args.grad_clip:
        #     scale_grad(model, args.grad_clip / grad_norm)

        # Save the model every 10 steps
        if step % 10 == 0:
            print(f"Saving model at step {step}")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'step': step,
            }
            torch.save(checkpoint, f"dit_checkpoint.pth")

        opt.step()
        update_ema(model_ema, model)

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    print("Done!")


if __name__ == '__main__':
    main()
