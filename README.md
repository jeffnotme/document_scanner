# Intelligent Document Scanner

A Python + OpenCV script that detects a document in a photo, applies a perspective warp to straighten it, and outputs a clean, sharpened scan.

## Features
- Automatic document edge detection (Canny + dilation/erosion)
- Perspective correction to A4-like aspect ratio
- Sharpening filter for readability
- Fallback: uses the whole image if no document is detected

## Requirements
- Python 3.8+
- See `requirements.txt`

## Installation
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Usage: python idoc\_scanner.py <input image> <output image>

