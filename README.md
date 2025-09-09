# AI-vs-Human-Text
### 
# HumAIn Detector 

A lightweight NLP project to detect whether a given text is **AI-generated** or **human-written**.  
Built with **PyTorch**, **Hugging Face Transformers**, and a simple **Gradio** interface.  

---

##  Project Goal

The rise of AI-generated content (ChatGPT, LLaMA, Claude, etc.) makes it increasingly important to **distinguish AI-written text from human writing**.  
This project aims to:

- Train and fine-tune a **BERT-based classifier** (starting with `prajjwal1/bert-tiny` for speed).  
- Provide a simple **web interface** (Gradio) for real-time text classification.  
- Support both **Turkish and English text**, since the dataset includes multi-language samples.  

---

##  Tech Stack

- **Python 3.11**
- **PyTorch** – deep learning backend  
- **Hugging Face Transformers** – model & tokenizer  
- **Datasets & Evaluate** – dataset handling and evaluation metrics  
- **Gradio** – interactive web UI  

---

##  Project Structure
<img width="451" height="297" alt="image" src="https://github.com/user-attachments/assets/66129aff-cd5d-4cc6-94a6-d536cf77f722" />


##
---

##  Training

The model was fine-tuned on a dataset of **~20k balanced samples** of AI vs. human text.  
Steps:
1. Preprocessing with `src/data/preprocess.py`
2. Fine-tuning with `src/models/train_bert.py`
3. Saving the trained model into `models/bert_tiny/`

Training config:
- **Base model:** `prajjwal1/bert-tiny`
- **Epochs:** 3  
- **Batch size:** 16  
- **Learning rate:** 2e-5  
- **Max length:** 128 tokens  

---

##  Evaluation

On the validation set (~4,000 samples):

- **Accuracy:** ~95%  
- **Macro F1:** ~94%  
- Balanced precision/recall across both `human` and `ai` labels.  

Example confusion matrix:

| True / Pred | Human | AI   |
|-------------|-------|------|
| Human       | 2413  | 75   |
| AI          | 129   | 1383 |

---

##  Deployment

The app is deployed on **Hugging Face Spaces**:  
 [Live Demo](https://huggingface.co/spaces/kuubraucar1/AiHuman)

---------------
Run locally:
#### bash
#### pip install -r requirements.txt
#### python app.py
#### Then open: http://127.0.0.1:7860
---------------

# Future Work

Fine-tune larger multilingual models (XLM-RoBERTa, BERT-base).

Expand dataset with more Turkish data.

Add confidence scores and explainability (e.g., attention visualization).

Integrate with FastAPI/Flask backend for production use.

# License

MIT License – feel free to use, modify, and distribute.



