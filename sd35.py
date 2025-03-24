import torch
from diffusers import StableDiffusion3Pipeline

# Set torch compute dtype
torch.set_default_dtype(torch.float16)  # Use regular float16 instead of bfloat16

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.float16,  # Changed from bfloat16 to float16
    variant="fp16"  # Explicitly request fp16 variant
)
pipe = pipe.to("cuda")

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("capybara.png")
