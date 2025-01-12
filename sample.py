import datetime
import os
import random

import numpy as np
import torch
import transformers
from diffusers import AutoencoderKL
from tqdm import tqdm
from transformers.image_transforms import to_pil_image

from dit import NextDiT
from transport import ODE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "dit_checkpoint.pth"


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

        text_inputs.to(device)
        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


def main(captions, size, num_sampling_steps, solver="euler", time_shifting_factor=1.0, do_extrapolation=False,
         cfg_scale=4.0):
    model = NextDiT(use_flash_attn=False, qk_norm=True, n_layers=4, cap_feat_dim=512, n_heads=8).to(device)

    tokenizer = transformers.T5Tokenizer.from_pretrained("google-t5/t5-small")
    tokenizer.padding_side = "right"

    # text_encoder
    text_encoder = transformers.T5EncoderModel.from_pretrained("google-t5/t5-small").to(device)

    # vae
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)

    # Check if the checkpoint file exists
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)

        # Load the model state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        # Optionally, load the step if you want to resume from that exact point
        start_step = checkpoint['step']

        model = model.to(device)

        print(f"Checkpoint loaded successfully, resuming from step {start_step}")
    else:
        print(f"Checkpoint {checkpoint_path} not found. Starting training from scratch.")

    model.eval()

    for idx, caption in tqdm(enumerate(captions)):
        caps_list = [caption]

        n = len(caps_list)
        w, h = int(size), int(size)
        latent_w, latent_h = w // 8, h // 8
        z = torch.randn([1, 4, latent_w, latent_h]).to(torch.float32).to(device)
        z = z.repeat(n * 2, 1, 1, 1)

        with torch.no_grad():
            cap_feats, cap_mask = encode_prompt([caps_list] + [""], text_encoder, tokenizer, 0.0)

        cap_mask = cap_mask.to(cap_feats.device)

        model_kwargs = dict(
            cap_feats=cap_feats,
            cap_mask=cap_mask,
            cfg_scale=cfg_scale
        )

        # if args.proportional_attn:
        #     model_kwargs["proportional_attn"] = True
        #     model_kwargs["base_seqlen"] = (train_args.image_size // 16) ** 2
        # else:
        #     model_kwargs["proportional_attn"] = False
        #     model_kwargs["base_seqlen"] = None
        model_kwargs["proportional_attn"] = False
        model_kwargs["base_seqlen"] = None

        # if do_extrapolation and args.scaling_method == "Time-aware":
        #     model_kwargs["scale_factor"] = math.sqrt(w * h / train_args.image_size ** 2)
        #     model_kwargs["scale_watershed"] = args.scaling_watershed
        # else:
        #     model_kwargs["scale_factor"] = 1.0
        #     model_kwargs["scale_watershed"] = 1.0
        model_kwargs["scale_factor"] = 1.0
        model_kwargs["scale_watershed"] = 1.0

        print(z.device)
        samples = ODE(num_sampling_steps, solver, time_shifting_factor).sample(
            z, model.forward_with_cfg, **model_kwargs
        )[-1]
        samples = samples[:1]

        factor = 0.18215
        samples = vae.decode(samples / factor).sample
        samples = (samples + 1.0) / 2.0
        samples.clamp_(0.0, 1.0)

        # Save samples to disk as individual .png files
        for i, (sample, cap) in enumerate(zip(samples, caps_list)):
            img = to_pil_image(sample.detach().numpy())
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
            save_path = f"images/{solver}_{num_sampling_steps}_{timestamp}_{i}.png"
            img.save(save_path)


if __name__ == '__main__':
    captions = ["dog"]
    size = 64
    sampling_steps = 20
    main(captions, size, sampling_steps)
