"""Wrapper to run app.py with TensorFlow disabled."""
import os

# Disable TensorFlow to avoid import errors
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

# Now import and run the app
from app import main

if __name__ == "__main__":
    main()
