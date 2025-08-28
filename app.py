import gradio as gr
import torch
import numpy as np
import spaces
from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwen_image_edit import QwenImageEditPipeline as QwenImageEditPipelineCustom
from optimization import optimize_pipeline_

pipe = None

def load_pipeline():
    """
    Loads and configures the Qwen-Image-Edit pipeline on GPU with custom scheduler, transformer, LoRA weights, and optimization.
    Returns:
        QwenImageEditPipelineCustom: The loaded and optimized pipeline.
    Raises:
        RuntimeError: If GPU is not available.
    """
    global pipe
    if pipe is not None:
        return pipe

    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for inference.")

    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": 1.0986,
        "max_image_seq_len": 8192,
        "max_shift": 1.0986,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": True,
        "time_shift_type": "exponential",
        "stochastic_sampling": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    pipe = QwenImageEditPipelineCustom.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        scheduler=scheduler,
        torch_dtype=dtype,
        cache_dir="/app/cache"
    ).to(device)

    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
    from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
    pipe.transformer.__class__ = QwenImageTransformer2DModel
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

    try:
        pipe.load_lora_weights(
            "/app/cache/hub",
            weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors"
        )
        pipe.fuse_lora()
    except Exception as e:
        print(f"Could not load LoRA: {e}")

    try:
        optimize_pipeline_(pipe, image=Image.new("RGB", (512, 512)), prompt="a")
    except Exception as e:
        print(f"AOT compile failed: {e}")

    return pipe

MAX_SEED = np.iinfo(np.int32).max

@gr.on(app="app.py", fn="infer")
@spaces.GPU(duration=60)
def infer(
    image: Image.Image,
    prompt: str,
    seed: int = 42,
    randomize_seed: bool = False,
    true_guidance_scale: float = 1.0,
    num_inference_steps: int = 8,
    num_outputs: int = 1,
):
    """
    Runs inference on the input image using the Qwen-Image-Edit pipeline.
    Args:
        image (Image.Image): Input image.
        prompt (str): Edit instruction.
        seed (int, optional): Random seed for reproducibility.
        randomize_seed (bool, optional): Whether to randomize the seed.
        true_guidance_scale (float, optional): Guidance scale for editing.
        num_inference_steps (int, optional): Number of inference steps.
        num_outputs (int, optional): Number of output images.
    Returns:
        tuple: (List of generated images, used seed)
    Raises:
        Exception: If inference fails.
    """
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipe = load_pipeline()

    try:
        outputs = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=" ",
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=num_outputs,
        ).images

        return outputs, seed
    except Exception as e:
        print(f"Inference error: {e}")
        raise e

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
"""

with gr.Blocks(css=css, title="Qwen-Image Edit") as demo:
    """
    Defines the Gradio UI for Qwen-Image Edit.
    """
    gr.HTML("""
    <div style="text-align:center; margin-bottom: 20px;">
        <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" width="400"/>
        <h2 style="color:#5b47d1;">Qwen-Image Edit (8-step Lightning)</h2>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
        output_gallery = gr.Gallery(label="Generated Images", columns=2, height="auto")

    with gr.Row():
        prompt = gr.Text(
            label="Edit Instruction",
            placeholder="E.g., 'Add a red sofa', 'Change text to 'Hello World''",
            show_label=False
        )
        run_button = gr.Button("Edit!", variant="primary")

    with gr.Accordion("Advanced Settings", open=False):
        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=42)
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

        with gr.Row():
            true_guidance_scale = gr.Slider(
                label="True Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0
            )
            num_inference_steps = gr.Slider(
                label="Inference Steps", minimum=4, maximum=28, step=1, value=8
            )
            num_outputs = gr.Slider(
                label="Number of Outputs", minimum=1, maximum=4, step=1, value=1
            )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[input_image, prompt, seed, randomize_seed, true_guidance_scale, num_inference_steps, num_outputs],
        outputs=[output_gallery, seed]
    )

if __name__ == "__main__":
    """
    Launches the Gradio demo server.
    """
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)