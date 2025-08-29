import runpod
import torch
import numpy as np
from PIL import Image
import base64
import io

pipe = None
COMPILED_MODEL_PATH = "compiled_pipe.pt"

def load_model():
    """
    Loads the pre-compiled pipeline from disk. This is very fast.
    """
    global pipe
    if pipe is not None:
        return pipe

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading compiled model from {COMPILED_MODEL_PATH} to {device}...")
    
    pipe = torch.load(COMPILED_MODEL_PATH)
    pipe.to(device)
    
    print("Model loaded successfully.")
    return pipe

def base64_to_pil(base64_string):
    """Decodes a base64 string into a PIL Image."""
    image_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def pil_to_base64(pil_image):
    """Encodes a PIL Image into a base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def handler(job):
    """
    RunPod serverless handler function.
    """
    global pipe
    if pipe is None:
        load_model()

    job_input = job['input']
    image_b64 = job_input.get('image')
    if not image_b64:
        return {"error": "Missing 'image' key in input. Please provide a base64-encoded image."}
        
    prompt = job_input.get('prompt', 'make it beautiful')
    seed = job_input.get('seed', None)
    true_guidance_scale = float(job_input.get('true_guidance_scale', 4.0))
    num_inference_steps = int(job_input.get('num_inference_steps', 8))

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    input_image = base64_to_pil(image_b64)

    PRESERVATION_PROMPT_SUFFIX = (
        "Strictly preserve all unmentioned objects, details, and the overall composition of the original image. "
        "This includes keeping background elements like doors, windows, and furniture exactly as they are."
    )
    final_prompt = f"{prompt}. {PRESERVATION_PROMPT_SUFFIX}"
    
    print(f"Processing job with seed: {seed}, steps: {num_inference_steps}, guidance: {true_guidance_scale}")
    print(f"Final prompt: {final_prompt}")

    try:
        output_image = pipe(
            image=input_image,
            prompt=final_prompt,
            negative_prompt="",
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=1,
        ).images[0]

        output_b64 = pil_to_base64(output_image)
        return {
            "image": output_b64,
            "seed": seed,
            "version": "1.0" 
        }

    except Exception as e:
        print(f"Inference error: {e}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})