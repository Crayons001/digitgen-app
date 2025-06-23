from flask import Flask
import os

def create_app():
    app = Flask(__name__)

    # Create image output directory if not exists
    os.makedirs("static/generated", exist_ok=True)

    from app.routes import main
    app.register_blueprint(main)

    return app
