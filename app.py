from flask import Flask, render_template, request
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from deoldify.visualize import DeOldify

app = Flask(__name__)

# Set up DeOldify colorization model
colorizer = DeOldify()

def apply_user_feedback(colorized_image, user_feedback):
    # Placeholder logic for applying user feedback
    # You need to implement the actual logic based on your requirements
    # For example, you can analyze user comments and adjust the colorized image accordingly
    # This might involve adjusting specific color channels or regions of the image

    # Here's a simple example: darken the entire image if the user provides feedback "Make it darker"
    if "darker" in user_feedback.lower():
        colorized_image -= 20

    return colorized_image

def colorize_image(input_image, user_feedback):
    # Convert byte data to OpenCV image
    img_np = np.frombuffer(input_image, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

    # Colorize the image using DeOldify
    colorized_img = colorizer.colorize(img)

    # Apply user feedback to the colorized image
    colorized_img = apply_user_feedback(colorized_img, user_feedback)

    # Convert colorized image to base64 for displaying in HTML
    image_pil = Image.fromarray(colorized_img)
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    colorized_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return colorized_image_base64

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded black and white image
        bw_image = request.files['bw_image'].read()

        # Get user feedback
        user_feedback = request.form['feedback']

        # Colorize the image based on the input and user feedback
        colorized_image = colorize_image(bw_image, user_feedback)

        return render_template('index.html', colorized_image=colorized_image)

    return render_template('index.html', colorized_image=None)

if __name__ == '__main__':
    app.run(debug=True)
