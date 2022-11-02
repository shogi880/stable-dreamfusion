import itertools
import math
import os
from contextlib import nullcontext
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import bitsandbytes as bnb
from argparse import Namespace


# Advanced settings for prior preservation (optional)
num_class_images = 12
sample_batch_size = 2
# `prior_preservation_weight` determins how strong the class for prior preservation should be 
prior_loss_weight = 1

# If the `prior_preservation_class_folder` is empty, images for the class will be generated with the class prompt. 
# Otherwise, fill this folder with images of items on the same class as your concept (but not images of the concept itself)
prior_preservation_class_folder = "./class_images"

# Teach the model the new concept (fine-tuning with Dreambooth)
class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_data,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data = instance_data # [B, H, W, 3]

        self.num_instance_images = len(self.instance_data)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(Path(class_data_root).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_data[index % self.num_instance_images].cpu()
        example["instance_images"] = self.image_transforms(Image.fromarray(np.array(instance_image * 255).astype(np.uint8)))
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        
        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def train_dreambooth(guidance, text, class_prompt, image_views, tmp_logs, max_train_steps=450):

    # Settings for your newly created concept
    # `instance_prompt` is a prompt that should contain a good description of what your object or style is, together with the initializer word `sks`
    # instance_prompt = f"modern disney style, sks {text}"
    instance_prompt = 'sks ' + text
    # Preserve class of the concept (e.g.: toy, dog, painting)
    prior_preservation = False
    prior_preservation_class_prompt = class_prompt

    # Setting up all training args
    args = Namespace(
        resolution=512,
        center_crop=True,
        instance_prompt=instance_prompt,
        learning_rate=5e-06,
        max_train_steps=max_train_steps,
        train_batch_size=1,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
        mixed_precision="no", # set to "fp16" for mixed-precision training.
        gradient_checkpointing=True, # set this to True to lower the memory usage.
        use_8bit_adam=True, # use 8bit optimizer from bitsandbytes
        seed=3434554,
        with_prior_preservation=prior_preservation,
        prior_loss_weight=prior_loss_weight,
        sample_batch_size=2,
        class_data_dir=prior_preservation_class_folder,
        class_prompt=prior_preservation_class_prompt,
        num_class_images=num_class_images,
        output_dir="dreambooth-concept",
    )

    # Generate Class Images
    # if(prior_preservation):
    if args.with_prior_preservation:
        class_images_dir = Path(prior_preservation_class_folder)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < num_class_images:
            num_new_images = num_class_images - cur_class_images
            print(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(prior_preservation_class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sample_batch_size)

            for example in tqdm(sample_dataloader, desc="Generating class images"):
                images = guidance.prompt_to_img(example["prompt"])

                for i, image in enumerate(images):
                    image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")
            with torch.no_grad():
                torch.cuda.empty_cache()
    
    tokenizer = guidance.tokenizer
    text_encoder = guidance.text_encoder
    vae = guidance.vae
    unet = guidance.unet
    
    for p in unet.parameters():
        p.requires_grad = True

    # Run training
    logger = get_logger(__name__)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    set_seed(args.seed)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        unet.parameters(),  # only optimize unet
        lr=args.learning_rate,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    
    train_dataset = DreamBoothDataset(
        instance_data=image_views,
        instance_prompt=args.instance_prompt,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_prompt=args.class_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # concat class and instance examples for prior preservation
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # Move text_encode and vae to gpu
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
  
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0


    for epoch in range(num_train_epochs):
        
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215

                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if args.with_prior_preservation:
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                    prior_loss = F.mse_loss(noise_pred_prior, noise_prior, reduction="none").mean([1, 2, 3]).mean()
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        
    # 
    # for p in unet.parameters():
    #     p.requires_grad = False
        
    guidance.unet = accelerator.unwrap_model(unet)
    guidance.vae = vae
    guidance.tokenizer = tokenizer
    guidance.text_encoder = text_encoder
    
    images = guidance.prompt_to_img(instance_prompt)
    Image.fromarray(images[0]).save(f'{tmp_logs}/end_stune_sd_tmp_{epoch}.png')
    