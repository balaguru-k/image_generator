# image_generator

1)Project Overview :

        This project is a simple AI Image Generator built using Streamlit, Diffusers, and Hugging Face models.Users can enter a text prompt, and the app generates an image using a pretrained Stable Diffusion model. The project is beginner-friendly and designed for both CPU and GPU systems.

2)Architecture :

        User Prompt → Streamlit UI → Diffusers Pipeline → Model (Stable Diffusion) → Generated Image
        
Components:

        * Frontend: Streamlit
        * Backend: Python
        * Model: Stable Diffusion (CompVis/stable-diffusion-v1-4)
        * Pipeline: Hugging Face Diffusers
        * Runtime: CPU/GPU

3)Setup & Installation :

* Clone the project:
  
         git clone https://github.com/your-username/your-repo.git
         cd your-repo
* Create virtual environment:
          python -m venv venv
* Activate the environment (Windows):
           venv\Scripts\activate
* Install dependencies:
            pip install -r requirements.txt
* Download the Stable Diffusion model:
  
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                      "CompVis/stable-diffusion-v1-4",
                        use_safetensors=True
             )
* Option 2 — Manual download (HuggingFace):
  
             * Go to https://huggingface.co/CompVis/stable-diffusion-v1-4
             * Accept the license
             * Download model files
             * Place inside models/stable-diffusion-v1-4/

4)Hardware Requirements

* CPU-Only
  
      * Minimum: 8 GB RAM
      * Recommended: 16 GB RAM
      * Images will take 20–60 seconds each

* GPU (Fast Generation)
  
      * NVIDIA GPU with 6 GB+ VRAM
      * CUDA installed
      * Generation time: 1–4 seconds
  
5)Usage Instructions:

      * Run the Streamlit app
              streamlit run app.py
      * Example Prompts
               "robot walking on fire"
               "robot walking on forest"
               "super hero fires the world"
               "A portrait of a robot in Van Gogh style"
               "earth without water"
          Generated images will appear on the screen.

6)Technology Stack:

         * Python 3.10+
         * Streamlit
         * Diffusers
         * Transformers
         * Torch (CPU/GPU)
         * Stable Diffusion v1.4
         
7)Prompt Engineering Tips:

          * Be specific:
                 “A realistic cat wearing sunglasses, 4K, detailed fur”
          * Add style keywords:
                 “cinematic lighting, ultra-detailed, cyberpunk style”
          * Add camera/lens info for realism:
                  “shot on 50mm lens, shallow depth of field”
          * Add artist/style references:
                  “in the style of Studio Ghibli”
          * Negative prompts (if supported):
                   “blurry, low quality, distorted face, extra limbs”

8)Limitations:

         * Slow on CPU (20–60 seconds per image)
         * High RAM usage
         * Cannot generate copyrighted character styles
         * Sometimes produces artifacts or inconsistent details
         * Limited to Stable Diffusion v1.4 capability

9)Future Improvements:

         * Fine-tuning on custom datasets
         * Support for additional models (FLUX, SDXL, DreamShaper)
         * Add image-to-image mode
         * Add style transfer
         * Add download history
         * Add GPU acceleration (CUDA/DirectML)
