# DeepSeek Fine-Tuning for Film Review Sentiment Analysis

This project demonstrates how to fine-tune the **DeepSeek LLM** using **Low-Rank Adaptation (LoRA)** and 4-bit quantization for memory-efficient training. The model is applied to analyze sentiment in film reviews, showcasing how AI can be adapted for specific tasks.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [References](#references)

---

## Features

1. **Fine-Tuning DeepSeek LLM**: Adapt the open-source DeepSeek model for sentiment analysis using LoRA.
2. **Memory Efficiency**: Utilize 4-bit quantization for efficient GPU usage.
3. **Custom Training Pipeline**: Train and evaluate the model on the IMDB dataset.
4. **Sentiment Prediction**: Generate predictions for custom text inputs.
5. **Result Archiving**: Automatically compress training results into a ZIP file.

---

## Installation

To get started, clone this repository and set up the environment:

```bash
# Clone the repository
git clone https://github.com/achuajays/Deepseek_Fine_Tunning.git

# Navigate to the project directory
cd Deepseek_Fine_Tunning/Film_Review

# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Setup

Before running the project, ensure the following:

1. **GPU Setup**: Make sure your system or cloud environment (e.g., Google Colab) has GPU support enabled.
   - In Colab, navigate to `Runtime > Change runtime type` and select `GPU`.
2. **Install Requirements**: Install all required Python packages using the provided `requirements.txt`.

---

## Usage

Run the main script to start the fine-tuning process and generate predictions:

```bash
python main.py
```

---

## File Structure

The repository is organized as follows:

```
Deepseek_Fine_Tunning/
│
├── Film_Review/
│   ├── main.py             # Main script for fine-tuning and prediction
│   ├── requirements.txt    # Python dependencies
│   ├── logs/               # Logs for training progress
│   └── README.md           # Project documentation (this file)
│

```

---

## References

- [DeepSeek LLM GitHub](https://github.com/deepseek-ai/DeepSeek-LLM)
- [DeepSeek LLM Paper on arXiv](https://arxiv.org/abs/2401.02954)
- [DeepSeek](https://www.deepseek.com/)

---

## Example Output

After running `main.py`, the script will:
1. Fine-tune the DeepSeek LLM using a subset of the IMDB dataset.
2. Generate sentiment predictions for sample film reviews:
   ```
   Input Review: The movie was absolutely fantastic! I loved the cinematography and the acting was superb.
   Predicted Sentiment: Positive
   ```

3. Compress training results into `results_archive.zip`.

---

Enjoy exploring and fine-tuning AI for sentiment analysis!
