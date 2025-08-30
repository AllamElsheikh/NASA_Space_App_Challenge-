# 🚀 Exoplanet AI Detector
*NASA International Space Apps Challenge 2025*


## 🌍 Project Overview

**Challenge:** Develop an AI/ML system to automatically analyze exoplanet survey data and identify potential exoplanets from NASA's treasure trove of space telescope observations.

Our solution leverages cutting-edge machine learning and deep learning techniques to process time-series photometric data from NASA's Kepler, K2, and TESS missions. By training on confirmed exoplanets and validated false positives, our AI system can identify the subtle periodic dimming patterns that indicate a planet transiting its host star.

### 🎯 **Mission Objectives**
- **Automate** exoplanet candidate identification from light curves
- **Enhance** detection accuracy beyond traditional methods  
- **Discover** potential new exoplanet candidates in archival data
- **Democratize** exoplanet research through accessible AI tools

---

## 📊 **Datasets**

We utilize open-source datasets from the **[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)**:

| Mission | Data Type | Time Coverage | Targets |
|---------|-----------|---------------|---------|
| **Kepler** | Long-cadence light curves | 2009-2013 | ~200,000 stars |
| **K2** | Short/long-cadence data | 2014-2018 | ~500,000 stars |
| **TESS** | 2-minute cadence | 2018-present | ~200,000 stars |

**Key Catalogs:**
- **KOI (Kepler Objects of Interest)** - 10,000+ labeled candidates
- **TOI (TESS Objects of Interest)** - 5,000+ recent discoveries
- **Confirmed Exoplanets** - Ground truth for training

---

## 🧠 **Methodology**

### **1. Data Pipeline**
```
Raw NASA Data → Quality Filtering → Feature Extraction → Model Training → Prediction
```

### **2. Machine Learning Approaches**
- **Classical ML**: Random Forest, XGBoost, LightGBM on engineered features
- **Deep Learning**: 1D CNN and RNN/LSTM on raw light curves
- **Ensemble Methods**: Voting and stacking classifiers for robust predictions

### **3. Feature Engineering**
- Transit depth and duration
- Orbital period detection
- Signal-to-noise ratio
- Stellar variability metrics
- Frequency domain analysis

---

## 📂 **Repository Structure**

```
exoplanet-ai-detector/
├── config/                   # Configuration files
│   ├── config.yaml          # Model hyperparameters
│   └── data_sources.yaml    # NASA API settings
├── src/                     # Source code
│   ├── data/               # Data processing modules
│   ├── models/             # ML/DL model definitions
│   └── utils/              # Utility functions
├── scripts/                # Automation scripts
├── tests/                  # Unit tests
├── data/                   # Datasets (gitignored)
│   ├── raw/               # Original NASA data
│   ├── processed/         # Cleaned datasets
│   └── external/          # Third-party data
├── notebooks/              # Jupyter analysis notebooks
├── saved_models/           # Trained models
├── results/                # Outputs and visualizations
└── docs/                   # Documentation
```

---

## ⚙️ **Quick Start**

### **Installation**
```bash
# Clone the repository
git clone https://github.com/your-username/exoplanet-ai-detector.git
cd exoplanet-ai-detector

# Install dependencies
make install
# or manually: pip install -r requirements.txt
```

### **Download Data**
```bash
# Download NASA catalogs and sample light curves
make download-data
```

### **Train Models**
```bash
# Train CNN model
python src/train.py --model cnn --epochs 50

# Train Random Forest
python src/train.py --model rf --config config/config.yaml
```

### **Evaluate Performance**
```bash
# Evaluate trained model
python src/evaluate.py --model saved_models/best_cnn_model.pkl
```

---

## 📈 **Results & Performance**

### **Model Performance**
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | TBD% | TBD% | TBD% | TBD |
| 1D CNN | TBD% | TBD% | TBD% | TBD |
| Ensemble | TBD% | TBD% | TBD% | TBD |

### **Key Findings**
- 🔍 **New Candidates**: X potential exoplanets identified in archival data
- 📊 **Detection Rate**: X% improvement over baseline methods
- ⚡ **Processing Speed**: X light curves analyzed per second

*Results will be updated as the project progresses during the hackathon.*

---

## 🛠️ **Technologies Used**

**Machine Learning:**
- scikit-learn, XGBoost, LightGBM
- PyTorch, TensorFlow/Keras

**Astronomy Tools:**
- Lightkurve (NASA light curve analysis)
- Astropy (astronomical computations)
- AstroQuery (NASA archive access)

**Data Science:**
- NumPy, Pandas, Matplotlib, Seaborn
- Jupyter Notebooks for exploration

---

## 👨‍🚀 **Team**

**[Your Team Name]**
- **[Name 1]** - Data Scientist & ML Engineer
- **[Name 2]** - Astronomer & Domain Expert  
- **[Name 3]** - Software Engineer & DevOps
- **[Name 4]** - UI/UX Designer (if applicable)

*Passionate space enthusiasts competing in NASA Space Apps Challenge 2025*

---

## 🌟 **Impact & Future Work**

### **Potential Impact**
- **Accelerate** exoplanet discovery pipeline
- **Enable** citizen science participation in space research
- **Support** NASA's quest to find Earth-like worlds

### **Next Steps**
- [ ] Real-time processing of TESS data streams
- [ ] Integration with NASA's exoplanet validation pipeline
- [ ] Web application for public use
- [ ] Mobile app for educational outreach

---

## 📚 **Documentation**

- **[Methodology](docs/methodology.md)** - Detailed technical approach
- **[API Reference](docs/api_reference.md)** - Code documentation
- **[Results Analysis](docs/results.md)** - Performance evaluation

---

## 🤝 **Contributing**

This project was developed during the NASA Space Apps Challenge 2025. We welcome feedback and contributions from the space science community!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **NASA** for providing open access to exoplanet survey data
- **NASA Space Apps Challenge** for inspiring global innovation
- **Kepler, K2, and TESS** mission teams for their groundbreaking work
- **Open-source astronomy community** for amazing tools and libraries

---


*"The cosmos is within us. We are made of star-stuff. We are a way for the universe to know itself."* - Carl Sagan

**Built with ❤️ for NASA Space Apps Challenge 2025**
