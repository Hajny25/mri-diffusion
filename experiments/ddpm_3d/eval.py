import math
from accelerate import Accelerator
import torch
from model import UNet3DModel
from diffusers import DDPMScheduler
from diffusers.utils import make_image_grid
from PIL import Image
from torchvision.transforms.functional import to_pil_image

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    #if images.ndim == 3:
    #    images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

run_id = "812e74f729884476b609c3252965ace0"
num_inference_timesteps=1000
num_slices=16
image_dims = (128, 160, 160)

accelerator = Accelerator(mixed_precision="fp16")

with accelerator.main_process_first():
    state_dict = torch.load(f"output/{run_id}/unet_weights.pth")
    model = UNet3DModel(in_channels=1)
    model.load_state_dict(state_dict)
    model.eval()

model = accelerator.prepare(model)
noise_scheduler = DDPMScheduler()
noise_scheduler.set_timesteps(num_inference_timesteps)
generator = torch.Generator(device=accelerator.device).manual_seed(134598)

vol = torch.randn((1, 1, *image_dims), device=accelerator.device)
with torch.no_grad():
    for t in noise_scheduler.timesteps:
        print(f"Timestep {t}/{num_inference_timesteps}")
        # 1. predict noise model_output
        t_up = t.unsqueeze(0).to(accelerator.device)
        model_output = model(vol, t_up)

        # 2. compute previous image: x_t -> x_t-1
        vol = noise_scheduler.step(model_output, t, vol, generator=generator).prev_sample


x = vol.squeeze(0).squeeze(0)
images = x[image_dims[0] - num_slices:image_dims[0] + num_slices]

images = images.detach().cpu()

pil_images = []
for img in images:
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    img = (img * 255).clamp(0, 255).byte()

    pil_images.append(
        Image.fromarray(img.numpy(), mode="L")
    )

grid_dims = int(math.sqrt(len(images)))
grid = make_image_grid(pil_images, rows=grid_dims, cols=grid_dims)
grid.save(f"Grid{accelerator.process_index}.png")



