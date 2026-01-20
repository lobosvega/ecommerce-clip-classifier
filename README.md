# AI Image Feed Classifier 🏍️

An automated pipeline that cleans, validates, and categorizes e-commerce product feeds using OpenAI's CLIP model.

## The Problem
E-commerce product feeds often contain broken links or mix "Lifestyle" images (bikes on the road) with "Product" images (parts on white backgrounds). Manual sorting is impossible for 10,000+ SKUs.

## The Solution
This tool:
1.  **Generates** potential image URLs using configurable suffixes (e.g., `_F.jpg`, `_ALT01.jpg`).
2.  **Validates** links by scraping them (removing 404s).
3.  **Classifies** valid images into "Product" vs "Lifestyle" buckets using semantic analysis (CLIP).

## How to Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Create a `config.json` (see `config.example.json`).
3.  Run the pipeline: `python pipeline.py`
