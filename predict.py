# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import requests
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "consChar_modified_min_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Hard-coded hacky function to download my custom weights (relies on requests import)
def download_my_weights():
    # Custom weights downloader
    custom_cpkt_url = "https://huggingface.co/JordanCo/realvisxlV40_v30InpaintBakedvae/resolve/main/realvisxlV40_v30InpaintBakedvae.safetensors"
    destination = "ComfyUI/models/checkpoints/realvisxlV40_v30InpaintBakedvae.safetensors"

    # Check if file exists
    if not os.path.exists(destination):
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Download custom weights
        print(f"Downloading custom weights from {custom_cpkt_url} to {destination}")
        try:
            with requests.get(custom_cpkt_url, stream=True, timeout=30) as response:
                response.raise_for_status()  # Check for HTTP errors
                with open(destination, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # Filter out keep-alive chunks
                            out_file.write(chunk)
            print("Download completed successfully.")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred while downloading: {e}")
            # Optionally delete the incomplete file
            if os.path.exists(destination):
                os.remove(destination)
    else:
        print(f"Custom weights already exist in {destination}")

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )
        # Quick Hack
        download_my_weights()

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        # Below is an example showing how to get the node you need and update the inputs

        positive_prompt = workflow["979"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["980"]["inputs"]
        negative_prompt["text"] = kwargs['negative_prompt']

        sampler = workflow["983"]["inputs"]
        sampler["seed"] = kwargs["seed"]

        imageLoader = workflow["991"]["inputs"]
        imageLoader["image"] = kwargs["image_filename"]

        prepCrop = workflow["1261"]["inputs"]
        prepCrop["crop_size_margin"] = kwargs["crop_size"]
        prepCrop["crop_pos_margin"] = kwargs["crop_margin"]

        pass

    def predict(
        self,
        prompt: str = Input(
            default="RAW photo of man age 28 with long blonde hair, out in nature, seeing the city, doing things",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="nsfw, nude, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name, trademark, watermark, title, multiple view, reference sheet, mutated hands and fingers, poorly drawn face, mutation, deformed, ugly, bad proportions, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, tatoo, amateur drawing, odd eyes, uneven eyes, unnatural face, uneven nostrils, crooked mouth, bad teeth, crooked teeth, photoshop, video game, censor, censored, ghost, b&w, weird colors, gradient background, spotty background, blurry background, ugly background, simple background, realistic, out of frame, extra objects, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of focus, blurry, very long body, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn eyes, cloned face, disfigured, deformed, cross-eye, extra limbs, missing limb, malformed hands, mutated, morbid, mutilated, disfigured, extra arms, extra hands, mangled fingers, contorted, conjoined, mismatched limbs, mismatched parts, bad perspective, black and white, oversaturated, undersaturated, bad shadow, cropped image, draft, grainy, pixelated",
        ),
        subject: Path = Input(
            description="Input photo of a person. A well-lit photo with a neutral expression and shoulder-up framing is recommended.",
            default=None,
        ),
        crop_size: int = Input(
            description="Output crop size relative to face. Larger values will be more \"zoomed out\" with worse likeness",
            default=2,
        ),
        crop_margin: int = Input(
            description="Output crop margin relative to face",
            default=0.5,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Make sure to set the seeds in your workflow
        seed = seed_helper.generate(seed)

        image_filename = None
        if subject:
            image_filename = self.filename_with_extension(subject, "image")
            self.handle_input_file(subject, image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_filename=image_filename,
            seed=seed,
            crop_size=crop_size,
            crop_margin=crop_margin,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
