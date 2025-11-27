import os
import json
from datetime import datetime
from typing import List, Optional

import torch
from diffusers import StableDiffusionPipeline


class ImageGenerator:
    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4", device: Optional[str] = None):
        self.device = "cpu"
        self.model_id = model_id
        self._load_pipeline()

    def _load_pipeline(self):
        print(f"Loading model {self.model_id} on CPU...")

        torch_dtype = torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            revision=None,
        )

        self.pipe.to("cpu")
        self.pipe.enable_attention_slicing()

    def build_prompt(self, prompt: str, style: str = "photorealistic", extra_descriptors: Optional[str] = None):
        descriptors = [prompt]

        if style:
            descriptors.append(style)

        if style == "photorealistic":
            descriptors.extend([
                "highly detailed",
                "sharp focus",
                "professional photography",
                "8k ultra realistic",
                "high clarity",
                "cinematic lighting"
            ])

        if extra_descriptors:
            descriptors.append(extra_descriptors)

        return ", ".join([d for d in descriptors if d])

    def generate(
        self,
        prompt: str,
        num_images: int = 1,
        style: str = "photorealistic",
        negative_prompt: Optional[str] = "blur, low quality, low resolution, distorted, bad anatomy, noisy, artifacts",
        guidance_scale: float = 8,
        num_inference_steps: int = 25, 
        output_dir: str = "outputs",
    ) -> List[str]:

        os.makedirs(output_dir, exist_ok=True)

        sd_prompt = self.build_prompt(prompt, style)
        generated_files = []

        for i in range(num_images):
            result = self.pipe(
                sd_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
            )

            image = result.images[0]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{output_dir}/{timestamp}_{i}.png"
            image.save(filename)

            metadata = {
                "prompt": prompt,
                "style": style,
                "full_prompt": sd_prompt,
                "negative_prompt": negative_prompt,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "model_id": self.model_id,
                "device": "cpu",
                "filename": filename,
                "timestamp": timestamp,
            }

            with open(filename + ".json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            generated_files.append(filename)

        return generated_files


if __name__ == "__main__":
    g = ImageGenerator()
    files = g.generate("a super hero fires the world", num_images=1)
    print("Saved:", files)
