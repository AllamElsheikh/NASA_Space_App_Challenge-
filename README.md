# NASA_Space_App_Challenge


# 🚀 Exoplanet AI Detector  
*NASA Space Apps Challenge 2025*

## 🌍 Overview  
This project was created for the **NASA International Space Apps Challenge 2025**.  
The challenge: **use AI/ML to automatically analyze exoplanet survey data and identify exoplanets**.  

We leverage open-source datasets from the **NASA Exoplanet Archive** (Kepler, K2, TESS) to build machine learning and deep learning models capable of classifying signals as *exoplanets* or *false positives*.  

---

## 📂 Repository Structure  

```markdown

exoplanet-ai-detector/
│
├── data/                     # Datasets (raw + processed)
│   ├── raw/                  # Original data from NASA archive
│   ├── processed/            # Cleaned & transformed data
│
├── notebooks/                # Jupyter/Colab notebooks (optional)
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/                      # Source code
│   ├── data_preprocessing.py
│   ├── model.py              # ML/DL model definition
│   ├── train.py              # Training pipeline
│   └── evaluate.py           # Model evaluation
│
├── saved_models/             # Trained models
│   └── best_model.pkl
│
├── results/                  # Outputs
│   ├── figures/              # Graphs, plots
│   └── reports/              # Evaluation metrics, logs
│
├── requirements.txt          # Dependencies
├── README.md                 # Project overview
├── LICENSE                   # License file
└── .gitignore                # Ignore unnecessary files

````

---

## 📊 Datasets  
We use datasets from the **[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)**:  
- **Kepler/K2 Light Curves** – time-series photometric data.  
- **TESS Light Curves** – ongoing mission data.  
- **KOI (Kepler Objects of Interest)** – labeled dataset with confirmed planets, false positives, and candidates.  

---

## 🧠 Approach  
1. **Data Preprocessing**  
   - Download light curves & KOI dataset  
   - Clean and normalize signals  
   - Create labeled training sets  

2. **Model Training**  
   - Classical ML (Random Forest, LightGBM) on KOI dataset  
   - Deep Learning (CNN/RNN) on raw light curves  

3. **Evaluation**  
   - Accuracy, Precision, Recall, F1-score  
   - ROC-AUC curve  
   - Confusion matrix  

4. **Deployment (optional)**  
   - Exoplanet classification demo via Flask or Streamlit app  

---

## ⚙️ Installation  

Clone the repo:  
```bash
git clone https://github.com/<your-username>/exoplanet-ai-detector.git
cd exoplanet-ai-detector
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train a model

```bash
python src/train.py --model cnn --epochs 20
```

### Evaluate a model

```bash
python src/evaluate.py --model saved_models/best_model.pkl
```

---

## 📈 Results

* Model accuracy: XX%
* Precision/Recall: XX% / XX%
* Discovered new candidate signals (if any)

---

## 👨‍🚀 Team

* Your Name(s)
* Role(s) (Data Scientist, ML Engineer, etc.)

---

## 📜 License

This project is licensed under the MIT License.

---

```

---

👉 Do you want me to also generate a **starter `requirements.txt`** (with libraries like `numpy`, `pandas`, `scikit-learn`, `tensorflow/torch`, etc.) so your repo is ready-to-run from the first commit?
```
