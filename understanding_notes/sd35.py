import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline
import os


def init_model(model_type):
    if model_type == "sd35":
        # Set torch compute dtype
        torch.set_default_dtype(torch.float16)  # Use regular float16 instead of bfloat16
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large",
            torch_dtype=torch.float16,  # Changed from bfloat16 to float16
            variant="fp16"  # Explicitly request fp16 variant
        )
    elif model_type == "flux":
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    
    return pipe

def sd35_inference(prompt):
    image = pipe(
        prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        ).images[0]
    return image

def flux_inference(prompt):
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image


model_type = "sd35"
pipe = init_model(model_type)
pipe = pipe.to("cuda")

save_dir = f"understanding_notes/{model_type}"
os.makedirs(save_dir, exist_ok=True)


# Intial prompt
intial_prompt = "a man sits on a motorcycle next to a very blue body of water"

# Dialogue
Dialogue = [
    "a man sits on a motorcycle next to a very blue body of water",
    "any animals? no",
    "can you see any people? yes",
    "is it a male or female? male",
    "i he facing the camera? yes",
    "is he happy? yes",
    "is he wearing a hat? no",
    "what color is his hair? brown",
    "is it daytime? yes",
    "is it sunny? yes",
    "can you see the sky? yes"
]
Dialogue = " ".join(Dialogue)

# Dialogue => Prompt Claude
D2P_claude = "A happy man with brown hair sitting on a motorcycle next to a \
    very blue body of water. The scene is outdoors during a sunny daytime, \
    with the sky visible in the background. The man is facing the camera. \
    There are no animals present in the image."


# Image => Text Claude 
I2T_claude_long = "A motorcyclist wearing protective gear including a white \
    helmet and blue/silver motorcycle jacket with black leather pants sits on \
    a light yellow/beige BMW F650 motorcycle. The rider is positioned on a \
    grassy area at the edge of a dramatic coastal cliff with rocky outcroppings \
    visible. Behind them is a stunning backdrop of deep blue water and steep \
    rocky cliffs. The scene appears to be taken during daylight with good \
    visibility, capturing the contrast between the motorcycle in the foreground \
    and the rugged natural landscape. The motorcycle's distinctive front fairing \
    and design elements are visible, and the rider appears to be gripping the \
    handlebars while stationary."

I2T_claude = "A motorcyclist in protective gear (white helmet, blue/silver \
    jacket, black pants) on a light yellow BMW F650 motorcycle. The rider \
    is positioned on grassy terrain at the edge of coastal cliffs with \
    dramatic blue water below. The scene showcases rugged rocky outcroppings \
    against the ocean backdrop, creating a striking contrast between the \
    motorcycle and the natural landscape."

# Claude_clarify
C2P_claude = "A motorcyclist on a yellow BMW adventure motorcycle sits on green grass at the edge of dramatic coastal cliffs. The rider wears a white helmet with tinted visor and a blue, black, and white motorcycle jacket with black pants. The motorcycle appears to be a BMW GS model with distinctive yellow bodywork and silver accents. In the background are spectacular steep cliff faces dropping directly into turquoise-blue ocean waters. The scene captures the motorcycle positioned on a grassy overlook with the stunning coastal landscape and sea stretching into the distance. The image shows excellent clarity with bright daylight illuminating both the rider and the breathtaking natural scenery."



# BLIP Captioning 1
blip_captioning_1 = "a photography of a man in a helmet and leather jacket riding a motorcycle"

# BLIP Captioning 2
# blip_captioning_2 =  "araffe on a yellow motorcycle on a dirt road near a body of water"

claude_35 = "The image depicts an urban street scene with a modern cityscape. There are several American flags hanging from decorative street lamps on the left side of the street. A tall blue glass high-rise building stands on the right side, and in the background, a white, distinctive architectural structure (likely a government or civic building) is visible. The street is lined with trees and has a clean, well-maintained appearance. A red vehicle can be seen driving down the street, and traffic lights and street signs are present. The sky is clear blue, suggesting it's a sunny day."

claude_35_short = "An urban street lined with American flags, featuring a modern blue glass high-rise, trees, and a white landmark building in the background. A red vehicle travels down the clean, well-maintained street under a bright blue sky.RetryClaude can make mistakes. Please double-check responses."


gemma3 = 'A vibrant, sun-drenched photograph captures a motorcyclist poised on a yellow BMW adventure bike overlooking a deep blue fjord. The rider is fully geared up in a black leather jacket with blue accents, black leather pants, and a white full-face motorcycle helmet with a'

prompt_dict = {
    # "1_intial_prompt": intial_prompt,
    # "2_Dialogue": Dialogue,
    # "3_D2P_claude": D2P_claude,
    # "4_I2T_claude": I2T_claude,
    # "5_C2P_claude": C2P_claude,
    # "6_blip_captioning_1": blip_captioning_1,
    # "7_cluade_35": claude_35,
    # "8_cluade_35_short": claude_35_short,
    "9_gemma3_1": gemma3,
}

for prompt_type in prompt_dict:
    prompt = prompt_dict[prompt_type]
    
    if model_type == "sd35":
        image = sd35_inference(prompt)
    elif model_type == "flux":
        image = flux_inference(prompt)
    image.save(f"{save_dir}/{prompt_type}.png")

