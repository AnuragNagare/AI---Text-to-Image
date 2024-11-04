import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the pre-trained Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

# Streamlit app
st.title("Stable Diffusion Image Generator")

# Text input for the prompt
prompt = st.text_input("Enter a description for the image:", "")

# Button to generate the image
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image... Please wait."):
            # Generate the image
            image = pipe(prompt).images[0]

            # Display the generated image
            st.image(image, caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please enter a description to generate an image.")

st.write("Enter a description and click 'Generate Image' to see your creation!")
