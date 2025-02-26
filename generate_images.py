import os
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer):
    loaded_learned_embeds = torch.load(learned_embeds_path)
    token = list(loaded_learned_embeds.keys())[0]
    embedding = loaded_learned_embeds[token]

    # Add the token in tokenizer
    num_added_tokens = tokenizer.add_tokens(token)
    i = 1
    while num_added_tokens == 0:
        print(f'Try adding token: {token} attempt: {i}')
        token = f"{token}{i}"
        num_added_tokens = tokenizer.add_tokens(token)
        i += 1

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Get the token id
    token_id = tokenizer.convert_tokens_to_ids(token)
    
    # Get the embeddings to modify
    text_encoder.get_input_embeddings().weight.data[token_id] = embedding
    return token

def main():
    st.title("Concept-Based Image Generation")

    # Initialize model
    @st.cache_resource
    def load_pipeline():
        pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2",
            torch_dtype=torch.float16
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        return pipeline

    pipeline = load_pipeline()
    
    # Load all concept embeddings
    concepts = [
        {"token": "<concept1>", "name": "Plane"},
        {"token": "<concept2>", "name": "Samurai"},
        {"token": "<concept3>", "name": "Hulk"},
        {"token": "<concept4>", "name": "Lego"},
        {"token": "<concept5>", "name": "Dolphin"}
    ]

    # UI for concept selection
    selected_concept = st.selectbox(
        "Select a concept",
        [concept["name"] for concept in concepts],
        index=0
    )

    # Get the corresponding token for the selected concept
    selected_token = next(concept["token"] for concept in concepts if concept["name"] == selected_concept)
    
    # Load the latest embedding for the selected concept
    embedding_dir = f"embeddings_{selected_token}"
    if os.path.exists(embedding_dir):
        embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.bin')]
        if embedding_files:
            latest_embedding = max(embedding_files, key=lambda x: int(x.split('-')[1].split('.')[0]))
            embedding_path = os.path.join(embedding_dir, latest_embedding)
            token = load_learned_embed_in_clip(
                embedding_path,
                pipeline.text_encoder,
                pipeline.tokenizer
            )

            # UI for prompt input
            prompt = st.text_input(
                "Enter your prompt",
                value=f"A photo of {token} in a garden",
                help="Enter a description of the image you want to generate"
            )

            # Generation parameters
            num_images = st.slider("Number of images", 1, 4, 1)
            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
            num_inference_steps = st.slider("Number of Inference Steps", 1, 100, 50)

            if st.button("Generate"):
                with st.spinner("Generating images..."):
                    images = pipeline(
                        prompt,
                        num_images_per_prompt=num_images,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps
                    ).images

                    # Display the generated images
                    cols = st.columns(num_images)
                    for idx, image in enumerate(images):
                        cols[idx].image(image, caption=f"Generated Image {idx+1}")
                        
                        # Add download button for each image
                        img_path = f"generated_{selected_concept}_{idx}.png"
                        image.save(img_path)
                        with open(img_path, "rb") as file:
                            cols[idx].download_button(
                                label="Download image",
                                data=file,
                                file_name=img_path,
                                mime="image/png"
                            )
        else:
            st.error(f"No embeddings found for {selected_concept}. Please train the model first.")
    else:
        st.error(f"No embedding directory found for {selected_concept}. Please train the model first.")

if __name__ == "__main__":
    main()