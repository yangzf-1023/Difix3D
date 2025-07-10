from pipeline_difix import DifixPipeline
from diffusers.utils import load_image

pipe = DifixPipeline.from_pretrained("/home/yangzhifan/Difix3D/checkpoint", local_files_only=True)
pipe.to("cuda")

input_image = load_image("/home/yangzhifan/4d-gaussian-splatting/output/N3V/cut_roasted_beef_vggt_woba_8e6/traj/ours_30000/renders/00010.png")
ref_image = load_image("/home/yangzhifan/4d-gaussian-splatting/data/N3V/cut_roasted_beef/images/cam00_0006.png")
prompt = "remove degradation"

output_image = pipe(prompt, image=input_image, ref_image=ref_image, num_inference_steps=1, timesteps=[199], guidance_scale=0.0).images[0]

output_image.save("ref_output.png")

