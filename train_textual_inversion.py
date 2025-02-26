import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import logging
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [path for path in Path(data_root).iterdir() if path.is_file()]
        self.num_images = len(self.image_paths)
        self._length = self.num_images * repeats

        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i % self.num_images])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        example = {"pixel_values": self.transforms(image)}
        text = self.placeholder_token
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        return example

def train_textual_inversion(concept_path, placeholder_token, initializer_token, output_dir):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
    )

    # Load the tokenizer and add the placeholder token
    tokenizer = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-2",
        subfolder="tokenizer",
    )
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # Load the Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=torch.float16,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder = pipeline.text_encoder
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # Freeze all parameters except for the token embeddings
    params_to_freeze = []
    for name, param in text_encoder.named_parameters():
        if "token_embedding" not in name:
            param.requires_grad = False
            params_to_freeze.append(param)

    # Create custom dataset
    train_dataset = TextualInversionDataset(
        data_root=concept_path,
        tokenizer=tokenizer,
        size=512,
        placeholder_token=placeholder_token,
        repeats=100
    )

    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),
        lr=5e-4,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-2
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * 100,
    )

    # Prepare everything with accelerator
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move VAE and UNet to same device and dtype as text encoder
    pipeline.vae = pipeline.vae.to(device=text_encoder.device, dtype=text_encoder.dtype)
    pipeline.unet = pipeline.unet.to(device=text_encoder.device, dtype=text_encoder.dtype)

    # Training loop
    for epoch in range(100):
        text_encoder.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = pipeline.vae.encode(batch["pixel_values"].to(dtype=text_encoder.dtype)).latent_dist.sample()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = pipeline.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        # Save the learned embeddings
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
            learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
            torch.save(learned_embeds_dict, os.path.join(output_dir, f"learned_embeds-{epoch}.bin"))

def main():
    # Define your 5 concepts here
    concepts = [
        {"path": "training_images/airplane", "token": "<concept1>", "init_token": "airplane"},
        {"path": "training_images/warrior", "token": "<concept2>", "init_token": "warrior"},
        {"path": "training_images/monster", "token": "<concept3>", "init_token": "monster"},
        {"path": "training_images/toy", "token": "<concept4>", "init_token": "toy"},
        {"path": "training_images/dolphin", "token": "<concept5>", "init_token": "dolphin"},
    ]

    # Train for each concept
    for concept in concepts:
        print(f"Training for concept: {concept['token']}")
        output_dir = f"embeddings_{concept['token']}"
        os.makedirs(output_dir, exist_ok=True)
        
        train_textual_inversion(
            concept_path=concept['path'],
            placeholder_token=concept['token'],
            initializer_token=concept['init_token'],
            output_dir=output_dir
        )

if __name__ == "__main__":
    main()