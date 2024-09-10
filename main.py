import modal

app = modal.App("moondream-mbot-tracker")

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "transformers",
        "einops",
        "Pillow",
        "torchvision",
        "opencv-python"
    )
    .run_commands(  # add flash-attn
        "pip install flash-attn==2.6.3 --no-build-isolation"
    )
)

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"

DEVICE = "cuda"
GPU = "L4"
BATCH_SIZE = 32

# GPU = "H100"
# BATCH_SIZE = 128

@app.function(gpu=GPU, image=image)
def run_moondream(video_url):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    import torch
    import time
    import requests
    import cv2
    import numpy as np
    from io import BytesIO

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
        trust_remote_code=True,
        revision=REVISION,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model.eval()

    # Download video
    response = requests.get(video_url)
    video_data = BytesIO(response.content)

    # Save video temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(video_data.getvalue())

    # Open video file
    cap = cv2.VideoCapture("temp_video.mp4")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Convert frames to PIL Images
    pil_images = []
    for frame in frames:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_images.append(pil_img)

    prompt = "Bounding box: mBot car robot"
    prompts = []
    for i in range(len(pil_images)):
        prompts.append(prompt)

    # Split the images into batches
    image_batches = [pil_images[i:i + BATCH_SIZE] for i in range(0, len(pil_images), BATCH_SIZE)]
    prompt_batches = [prompts[i:i + BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]

    # Initialize an empty list to store all answers
    all_answers = []

    # Process each batch
    for img_batch, prompt_batch in zip(image_batches, prompt_batches):
        batch_answers = model.batch_answer(
            images=img_batch,
            prompts=prompt_batch,
            tokenizer=tokenizer,
        )
        all_answers.extend(batch_answers)

    # Replace the original answers with all_answers
    answers = all_answers

    return [answers, pil_images[0].width, pil_images[0].height]

@app.function(image=image)
def calculate_distance_travelled(url):
    import time
    import json

    def calculate_bbox_distance(bbox1, bbox2):
        """
        Calculate the distance between two bounding boxes.

        :param bbox1: List containing [x1, y1, x2, y2] of the first bounding box
        :param bbox2: List containing [x1, y1, x2, y2] of the second bounding box
        :return: Float representing the distance between the two bounding boxes
        """
        import math

        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate centers of bounding boxes
        center_x1 = (x1_1 + x2_1) / 2
        center_y1 = (y1_1 + y2_1) / 2
        center_x2 = (x1_2 + x2_2) / 2
        center_y2 = (y1_2 + y2_2) / 2

        # Calculate the Euclidean distance between centers
        distance = math.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)

        return distance

    def calculate_bbox_distance_normalized(bbox1, bbox2, image_width, image_height):
        """
        Calculate the distance between two bounding boxes with normalized coordinates.

        :param bbox1: List containing normalized [x1, y1, x2, y2] of the first bounding box
        :param bbox2: List containing normalized [x1, y1, x2, y2] of the second bounding box
        :param image_width: Width of the image in pixels
        :param image_height: Height of the image in pixels
        :return: Float representing the distance between the two bounding boxes in pixels
        """
        # Denormalize coordinates
        bbox1_pixels = [
            bbox1[0] * image_width,
            bbox1[1] * image_height,
            bbox1[2] * image_width,
            bbox1[3] * image_height
        ]
        bbox2_pixels = [
            bbox2[0] * image_width,
            bbox2[1] * image_height,
            bbox2[2] * image_width,
            bbox2[3] * image_height
        ]

        # Use the original function to calculate distance in pixels
        return calculate_bbox_distance(bbox1_pixels, bbox2_pixels)

    start = time.time()
    result = run_moondream.remote(url)
    end = time.time()
    print(f"Total inference time (sec): {end - start}")

    answers = result[0]

    distance = 0
    for i in range(1, len(answers)):
        bbox1 = json.loads(answers[i - 1])
        bbox2 = json.loads(answers[i])
        distance += calculate_bbox_distance_normalized(bbox1, bbox2, result[1], result[2])

    return distance


@app.local_entrypoint()
def main(url):
    distance = calculate_distance_travelled.remote(url)
    print(f"Distance travelled: {distance} px")
