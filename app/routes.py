from flask import Blueprint, render_template, request
from app.utils import generate_images

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
    digit = None
    images = []

    if request.method == 'POST':
        digit = int(request.form['digit'])
        images = generate_images(digit, num_samples=5)

    return render_template('index.html', digit=digit, images=images)
