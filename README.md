# Text to Image Generator with Gradio and Stable Diffusion

This repository hosts a Gradio-based web application for generating images from text prompts using the Stable Diffusion model. The project leverages PyTorch and the `diffusers` library to load and run the model efficiently.

## Features

- User-friendly web interface built with Gradio.
- Generates high-quality images from text prompts.
- Supports GPU acceleration for faster generation (uses CUDA if available).
- Outputs single or multiple images based on user input.

## Prerequisites

Ensure that you have the following installed before running this project:

- Python 3.8 or later
- CUDA-enabled GPU (optional but recommended for better performance)
- `torch`, `transformers`, `diffusers`, `gradio`, `PIL`

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/owais-mazhar/text-to-image-generator.git
    cd text-to-image-generator
    ```

2. **Install dependencies**:
    Install the required Python libraries using `pip`:
    ```bash
    pip install torch torchvision torchaudio diffusers transformers gradio pillow
    ```

3. **Add an authentication token**:
    Create a file named `authtoken.py` in the project root and include your Hugging Face API token:
    ```python
    auth_token = "your_huggingface_auth_token"
    ```

## Usage

1. **Run the application**:
    ```bash
    python app.py
    ```

2. **Access the web app**:
    Open your web browser and go to `http://127.0.0.1:7860/` to use the application.

## Code Overview

### Model Loading

```python
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    token=auth_token, 
    low_cpu_mem_usage=True
)
pipe.to(device)
```

### Image Generation Function

```python
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
```

### Gradio Interface

```python
with gr.Blocks() as demo:
    gr.Markdown("# Text to Image Generator")
    prompt_input = gr.Textbox(label="Enter your prompt:")
    generate_button = gr.Button("Generate")
    output_image = gr.Image(label="Generated Image")
    generate_button.click(fn=generate_image, inputs=prompt_input, outputs=output_image)
demo.launch()
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the `diffusers` library.
- [Gradio](https://gradio.app/) for building interactive web applications easily.
- [CompVis](https://github.com/CompVis) for the Stable Diffusion model.

## Contact

For questions, feel free to reach out at [your-email@example.com] or open an issue in the repository.