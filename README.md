# Wheat Seed Classification Using ANN

This project is **Machine Learning Lab 6**, where an Artificial Neural Network (ANN) is used to classify wheat seeds into three classes from seven numerical kernel measurements.

## Work Performed

- Data inspection, missing-value and duplicate checks
- Exploratory analysis using bar charts, histograms, box plots, scatter plots, and a correlation heatmap
- Stratified 80:20 train-test split and `StandardScaler` preprocessing
- ANN with **7 input features, Dense(16), Dense(8), and Dense(3)** layers
- ReLU hidden activations, softmax output, Adam optimizer, and early stopping
- Accuracy, weighted precision, recall, F1-score, classification report, and confusion matrix

## Recorded Notebook Result

The supplied notebook records **85.71% test accuracy** on **42 test samples** (168 training samples). Its weighted precision is **86.25%** and weighted F1-score is **85.00%**. Training is stochastic, so fresh runs may give different scores.

## Project Structure

```text
wheat-seed-ann-classification/
├── data/
│   └── README.md
├── notebook/
│   └── Seed_Classification_ANN.ipynb
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Run

Install dependencies (Python 3.10–3.12 recommended):

```bash
pip install -r requirements.txt
```

Run the standalone script:

```bash
python main.py
```

Alternatively, first run `python main.py` once to prepare `data/seeds_new.csv`, then open `notebook/Seed_Classification_ANN.ipynb` in Jupyter and run its cells. The notebook reads `../data/seeds_new.csv` relative to its folder.

The original `seeds_new.csv` was **not included** with the uploaded notebook. The published notebook updates its CSV path and removes a demonstration-only test-set scaler refit to avoid inconsistent preprocessing. `main.py` loads `data/seeds_new.csv` if available; otherwise it attempts to download the corresponding public **UCI Seeds** dataset and save a compatible CSV.

**Dataset:** [UCI Seeds](https://archive.ics.uci.edu/dataset/236/seeds), Charytanowicz et al. (2010), [DOI 10.24432/C5H30K](https://doi.org/10.24432/C5H30K), CC BY 4.0.

## Author

**Shreyank Nipane**  
B.Tech CSE — Artificial Intelligence & Machine Learning
