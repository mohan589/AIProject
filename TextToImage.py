from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

MODEL_ID = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("mps")

PROMPT = "a photo of an astronaut riding a horse on mars"
image = pipe(PROMPT).images[0]
image.save('data/astronaut_riding_horse_on_mars.png')

PROMPT = "a photo of an iron man as saint"
image = pipe(PROMPT).images[0]
image.save('data/iron_man_as_saint.png')