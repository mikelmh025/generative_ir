import sys
sys.path.append('./Infinity')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import random
import torch
# torch.cuda.set_device(2)
import cv2
import numpy as np
from tools.run_infinity import *

model_path='temp/weights/infinity_model/infinity_8b_weights'
vae_path='temp/weights/infinity_model/infinity_vae_d56_f8_14_patchify.pth'
text_encoder_ckpt = 'temp/weights/flan-t5-xl'
args=argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=14,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_8b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=1,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type='torch_shard',
    seed=0,
    bf16=1,
    save_file='tmp.jpg'
)


# load text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# load vae
vae = load_visual_tokenizer(args)
# load infinity
infinity = load_transformer(vae, args)


# prompt = """a cat holds a board with the text 'diffusion is dead'"""
# prompt = """Interior view of a car, side profile, woman seated in driver's seat working on two laptops. Woman with dark hair pulled back in a bun, wearing a black blazer and beige pants, focused expression, typing on a silver laptop computer. A second monitor is mounted to the left of the primary laptop. Car interior features tan leather bucket seats, center console, and beige dashboard. Visible through the open window: industrial building exterior, gray siding, loading dock area, blue Chrysler Sebring parked outside, yellow cart nearby. Natural lighting, overcast day. Medium shot, shallow depth of field, slightly grainy texture. Realistic photography style."""
# prompt = """Close-up shot of a grilled panini sandwich cut open, revealing melted cheese and chopped vegetables inside. The bread has a dark, crispy crust with visible grill marks and toasted texture. Inside, the filling consists of vibrant green zucchini and spinach mixed with creamy white mozzarella cheese, all nestled between layers of the rustic bread. A single fork rests near the sandwich on a simple white plate. The background is blurred, suggesting a cafe setting with hints of wooden furniture and a citrus fruit (orange) out of focus. Warm lighting highlights the textures and colors, creating a cozy and appetizing feel. Shallow depth of field emphasizes the sandwich's detail. High resolution, food photography, natural light."""

prompt="""A bustling marketplace scene in Marrakech, Morocco, bathed in bright sunlight. A heavily laden cart dominates the foreground, piled high with numerous orange plastic bottles filled with juice, secured by twine. The cart has green wheels and a faded olive-green frame. Pedestrians, dressed in casual clothing—jackets, sweaters, jeans—move through the square, some looking directly at the camera. In the background, traditional Moroccan architecture rises: ochre-colored buildings with intricate white detailing, including a prominent minaret with ornate patterns. A rooftop terrace with dark awnings sits atop one building. A small handcart is visible to the left, partially obscured by people. The sky is clear blue. The overall impression is lively and vibrant, capturing the essence of a busy market day. Medium shot, natural lighting, documentary photography style."""


cfg = 3
tau = 1.0
h_div_w = 1/1 # aspect ratio, height:width
seed = random.randint(0, 10000)
enable_positive_prompt=0

h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
generated_image = gen_one_img(
    infinity,
    vae,
    text_tokenizer,
    text_encoder,
    prompt,
    g_seed=seed,
    gt_leak=0,
    gt_ls_Bl=None,
    cfg_list=cfg,
    tau_list=tau,
    scale_schedule=scale_schedule,
    cfg_insertion_layer=[args.cfg_insertion_layer],
    vae_type=args.vae_type,
    sampling_per_bits=args.sampling_per_bits,
    enable_positive_prompt=enable_positive_prompt,
)
args.save_file = 'ipynb_tmp.jpg'
os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
cv2.imwrite(args.save_file, generated_image.cpu().numpy())
print(f'Save to {osp.abspath(args.save_file)}')