import base64
from PIL import Image
import cv2
from flask import Flask, jsonify, render_template, request
import torch.serialization
import functools
from fastai.basic_train import Recorder
from fastai.callbacks.hooks import Hook, _hook_inner, Hooks
from fastai.layers import (
    MergeLayer, SigmoidRange, SelfAttention,
    PixelShuffle_ICNR, SequentialEx
)
from fastai.torch_core import ParameterModule
from deoldify.unet import (
    DynamicUnetDeep, UnetBlockDeep, CustomPixelShuffle_ICNR
)
from torch.nn.utils.spectral_norm import SpectralNorm, SpectralNormStateDictHook, SpectralNormLoadStateDictPreHook
from torch.nn.utils.weight_norm import WeightNorm
from torch.optim import Adam
from torch.nn import functional as F
import torchvision.models.resnet as resnet
import pathlib
import torch.nn as nn

safe_globals = [
    functools.partial,
    Recorder,
    Hook,
    _hook_inner,
    Hooks,
    MergeLayer,
    SigmoidRange,
    SelfAttention,
    PixelShuffle_ICNR,
    SequentialEx,
    ParameterModule,
    DynamicUnetDeep,
    UnetBlockDeep,
    CustomPixelShuffle_ICNR,
    SpectralNorm,
    SpectralNormStateDictHook,
    SpectralNormLoadStateDictPreHook,
    WeightNorm,
    Adam,
    resnet.BasicBlock,
    pathlib.PosixPath,
    slice,
    getattr,
    nn.Sequential,
    nn.Conv2d,
    nn.Conv1d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.AvgPool2d,
    nn.MaxPool2d,
    nn.ReplicationPad2d,
    nn.PixelShuffle,
    nn.ModuleList,
    nn.modules.module._WrappedHook,
    torch.utils.hooks.RemovableHandle,
    F.l1_loss,
]

torch.serialization.add_safe_globals(safe_globals)
from deoldify.visualize import get_image_colorizer
import numpy as np

app = Flask(__name__)

# Download model on-the-fly if it doesn't exist
MODEL_DIR = pathlib.Path("models")
MODEL_PATH = MODEL_DIR / "ColorizeArtistic_gen.pth"

def ensure_model():
    MODEL_DIR.mkdir(exist_ok=True)
    if not MODEL_PATH.exists():
        import requests
        print("Downloading model...")
        url = "https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth"
        response = requests.get(url, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded...")

ensure_model()


# Load the colorizer
colorizer = get_image_colorizer(artistic=True, root_folder=pathlib.Path('.'))

def apply_user_feedback(image: np.ndarray, feedback: str) -> np.ndarray:
    # Example: darken image if feedback contains "darker"
    if "darker" in feedback.lower():
        return np.clip(image - 20, 0, 255).astype(np.uint8)
    return image

def colorize_image(input_image_bytes: bytes, feedback: str) -> str:
    # Save uploaded image temporarily
    input_path = pathlib.Path("temp_bw.jpg")
    with open(input_path, 'wb') as f:
        f.write(input_image_bytes)

    # Run DeOldify
    result = colorizer.get_transformed_image(
        str(input_path),
        render_factor=35,
        post_process=True
    )

    print(f"[DEBUG] Result from colorizer: {type(result)} - {result}")

    # Handle if result is a PIL Image
    if isinstance(result, Image.Image):
        result_img = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    else:
        result_img = cv2.imread(str(result))

    if result_img is None:
        raise ValueError("Colorization failed, result image is None.")

    if feedback:
        result_img = apply_user_feedback(result_img, feedback)

    # Convert to base64 for web rendering
    _, buffer = cv2.imencode('.jpg', result_img)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    return base64_img


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            bw_image = request.files['bw_image'].read()
            feedback = request.form.get('feedback', '')
            colorized_image = colorize_image(bw_image, feedback)
            return jsonify({"image": colorized_image})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)

