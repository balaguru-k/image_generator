import streamlit as st
from generate import ImageGenerator
import os


st.set_page_config(page_title="AI Image Generator", layout="centered")

st.title("AI-Powered Image Generator")
st.write("Open-source Stable Diffusion demo — runs locally. GPU recommended.")


with st.sidebar:
    st.header("Settings")
    model_id = st.text_input("Model ID (HuggingFace)", value="CompVis/stable-diffusion-v1-4")
    device_pref = "cuda" if st.checkbox("Use GPU (if available)", value=True) else "cpu"
    num_steps = st.slider("Inference steps", 10, 50, 30)
    guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5)


prompt = st.text_area("Prompt", value="A portrait of a robot in Van Gogh style")
style = st.selectbox("Style", ["photorealistic", "artistic", "cartoon", "sketch"])
num_images = st.number_input("Number of images", min_value=1, max_value=4, value=1)
neg_prompt = st.text_input("Negative prompt (optional)", value="")
extra = st.text_input("Extra descriptors (optional)", value="")

if st.button("Load model & Generate"):
    with st.spinner("Loading model and generating, this may take a minute..."):
        gen = ImageGenerator(model_id=model_id, device=device_pref)
        files = gen.generate(
            prompt=prompt,
            num_images=num_images,
            style=style,
            negative_prompt=neg_prompt if neg_prompt.strip() else None,
            guidance_scale=guidance,
            num_inference_steps=num_steps,
            output_dir="outputs",
        )


    st.success("Generation complete")
    for fpath in files:
        st.image(fpath, use_column_width=True)
        with open(fpath, "rb") as f:
            btn = st.download_button(
                label="Download image",
                data=f,
                file_name=os.path.basename(fpath),
                mime="image/png",
       )


st.markdown("---")
st.info("Tip: If you run into memory errors on GPU, reduce `num_images`, `num_inference_steps`, or switch to CPU mode.")