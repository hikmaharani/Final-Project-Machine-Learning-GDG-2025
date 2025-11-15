## Named Entity Recognition (NER) Bahasa Indonesia - Model XLM-RoBERTa

## Project Summary

This project implements a **Named Entity Recognition (NER)** model for the Indonesian language using the **XLM-RoBERTa** architecture.

The core goal was to build a robust model by implementing techniques to mitigate class imbalance, including:
1.  **Focal Loss** and **Class Weighting** in the training objective.
2.  **Data Augmentation** on the training dataset.

The model classifies three main entity types: **PERSON**, **PLACE**, and **ORGANISATION**. The model is deployed via a **Flask web application**.

---

##  Model Performance (Test Set)

The final model performance metrics, evaluated on the dedicated test set using **sequence-level F1 score** (seqeval):

| Metric | Score (Micro Average) |
| :--- | :---: |
| **F1 Score** | **0.8280** |
| **Precision** | 0.8096 |
| **Recall** | 0.8473 |

### Detailed F1 Score
| Entity | F1-Score |
| :--- | :---: |
| **PERSON** | 0.85 |
| **PLACE** | 0.87 |
| **ORGANISATION** | 0.69 |

---

##  Project Structure

---

NER-FLASK-APP/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── model_utils.py
├── train_model.py

├── data/
│   └── (file data NERgrit)
├── models/
│   └── ner_xlm_roberta_final/
│       └── (semua file model seperti model.safetensors, tokenizer.json, dll.)

├── notebooks/
│   └── train_ner_model.ipynb
├── results/
│   └── (log, checkpoint lama)

├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html
└── venv/

##  How to Run the Deployment (Flask)

### Prerequisites
1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPO_LINK>
    cd NER-FLASK-APP
    ```
2.  **Setup Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
1.  Ensure the trained model files (e.g., `pytorch_model.bin`, `config.json`, `tokenizer.json`) are correctly placed within the `models/` directory or loaded via `model_utils.py`.
2.  Run the Flask application:
    ```bash
    python app.py
    ```
3.  The application will be available at the URL displayed in the console (typically **http://127.0.0.1:5000**).

---

## 🛠️ Usage & Functional Deployment

### Actual Testing
The Flask application is designed to meet the deployment criteria:
* **Running:** The `app.py` script starts successfully.
* **Real-time Inference:** A user can input a new, unseen sentence on the web interface (`index.html`).
* **Output:** The application uses **`model_utils.py`** to process the input and displays the correct Named Entities detected (**PERSON**, **PLACE**, **ORGANISATION**).