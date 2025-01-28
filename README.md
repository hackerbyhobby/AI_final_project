
# SMiShing & Scam Detector with Safe Browsing

## Overview
This project is an AI-powered tool designed to detect SMiShing (SMS phishing) and other scam messages. Leveraging cutting-edge machine learning and natural language processing (NLP), it classifies input as "SMiShing," "Other Scam," or "Legitimate." Additionally, the tool integrates with Google's Safe Browsing API to analyze URLs for potential threats.

## Key Features
- **Multi-language Support**: Detects and processes English and Spanish messages with language-specific keyword analysis.
- **Zero-shot Classification**: Uses the `joeddav/xlm-roberta-large-xnli` model to classify messages without requiring retraining.
- **URL Threat Analysis**: Integrates with the Google Safe Browsing API to check URLs for malware, phishing, and other threats.
- **OCR Integration**: Extracts text from images using Tesseract OCR, enabling message classification from screenshots.
- **Explainability**: Provides SHAP-based visual explanations for model predictions.
- **Interactive Interface**: Built with Gradio for an intuitive user interface.

## Technologies Used
- **Hugging Face Transformers**: For zero-shot text classification.
- **Tesseract OCR**: To extract text from images.
- **SHAP (SHapley Additive ExPlanations)**: For model interpretability.
- **Google Safe Browsing API**: To detect malicious URLs.
- **Gradio**: For an interactive user interface.
- **LangDetect & Deep Translator**: For language detection and translation.

## Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Tesseract OCR is installed:
   - **Linux**: `sudo apt install tesseract-ocr`
   - **MacOS**: `brew install tesseract`
   - **Windows**: [Install Tesseract](https://github.com/tesseract-ocr/tesseract)

4. Set your Google Safe Browsing API key as an environment variable:
   ```bash
   export SAFE_BROWSING_API_KEY=<your-api-key>
   ```

## Usage
1. Run the application:
   ```bash
   python app.py
   ```
2. Open the provided Gradio interface in your browser.
3. Paste a suspicious SMS message or upload a screenshot to analyze it.

## File Structure
- **app.py**: Main application script.
- **smishing_keywords.txt**: Keywords for identifying SMiShing messages.
- **other_scam_keywords.txt**: Keywords for identifying other scams.

## Screenshots
![Interface Screenshot](https://via.placeholder.com/600x400.png?text=Add+Your+Screenshot+Here)

## Contributing
Feel free to submit issues or feature requests. Contributions are welcome!

## License
This project is licensed under the [MIT License](LICENSE).
