
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

### 🏃 Usage Guide
1. Run the application:
   ```bash
   python app.py
   ```
2. Open the provided Gradio interface in your browser.

1. **Select Input Type**
   - **Text**: Paste or type a suspicious SMS (English or Spanish).
   - **Screenshot**: Upload an image or screenshot containing the suspicious text.

2. **Classify**
   - The system performs OCR (if a screenshot is uploaded) and automatically detects the language.
   - Probability is boosted if certain keywords (or a URL) appear.
   - The final label (`SMiShing`, `Other Scam`, or `Legitimate`) and confidence are displayed, along with any detected keywords/URLs.

3. **Interpret Results**
   - High **SMiShing**: The message strongly resembles phishing attempts.
   - High **Other Scam**: Indicates a scam that is not strictly phishing.
   - **Legitimate**: Suggests no malicious indicators were found.

## File Structure
- **app.py**: Main application script.
- **smishing_keywords.txt**: Keywords for identifying SMiShing messages.
- **other_scam_keywords.txt**: Keywords for identifying other scams.

## Screenshots
![Interface Screenshot](https://via.placeholder.com/600x400.png?text=Add+Your+Screenshot+Here)

### 🔧 Customization

- **Keywords**
  - Edit or add lines in `smishing_keywords.txt` or `other_scam_keywords.txt` to reflect new terms.
  - The app auto-detects these keywords (and translates to Spanish if necessary).

- **Boost Amount**
  - Currently set to specific multipliers (e.g., `0.30`, `0.35`).
  - Adjust these values in the `boost_probabilities(...)` function.

- **Language Detection**
  - By default, the app uses `langdetect`.
  - You can swap it for a more advanced language detection library if needed.

- **OCR**
  - If you prefer a different OCR engine (e.g., EasyOCR or Gemini-based OCR), replace the Tesseract calls with your desired approach.
 
### 💡 Disclaimers & Warnings

- **False Positives/Negatives**: No ML-based system is perfect. Always verify suspicious messages.
- **Privacy**: We recommend not storing user data or images unless absolutely required.
- **External Dependencies**: The performance and accuracy rely on Tesseract’s OCR, Hugging Face Transformers, and external translation libraries.

## Contributing
Feel free to submit issues or feature requests. Contributions are welcome!

## License
This project is licensed under the [MIT License](LICENSE).
