import gradio as gr
from PIL import Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from authtoken import auth_token
# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline with half-precision to reduce VRAM usage
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,  # Use half-precision
    token=auth_token, 
    low_cpu_mem_usage=True
)

pipe.to(device)

def generate_image(prompt):
    if prompt:
        with autocast(device):
            try:
                image = pipe(prompt, guidance_scale=8.5)["images"][0]
                return image
            except RuntimeError as e:
                return f"An error occurred: {e}"
    else:
        return "Please enter a prompt."

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Text to Image Generator")
    
    prompt_input = gr.Textbox(label="Enter your prompt:")
    generate_button = gr.Button("Generate")
    output_image = gr.Image(label="Generated Image")

    generate_button.click(fn=generate_image, inputs=prompt_input, outputs=output_image)

# Launch the Gradio app
demo.launch()