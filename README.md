# 🏭 AI-Driven Manufacturing Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)

An **AI Batch Optimization System** built to optimize manufacturing processes by balancing quality outputs (Hardness, Dissolution Rate, Content Uniformity) with sustainability metrics (Carbon Emissions). 

The platform predicts outcomes based on live sensor data and actively suggests optimal machine parameter tweaks to maintain high performance while strictly pursuing a minimal carbon footprint.

## ✨ Core Features

* **Multi-Target Prediction Engine**: Utilizes Scikit-Learn’s `MultiOutputRegressor` stacked with a `RandomForestRegressor` to predict multiple simultaneous output qualities based on batch variables.
* **Energy Anomaly Detection**: Automatically flags power spikes outside normal standard deviation controls via real-time statistical boundary detection.
* **Smart Parameter Optimization**: Iterates via a simulated random walk algorithm (`np.random.normal`) to find the best configuration changes (Golden Signature) that yield the highest possible combined quality/sustainability score.
* **Interactive Web Interface**: A beautifully styled, dark-mode Streamlit dashboard (`app2.py`) loaded with KPI cards, dynamic charts, and an interactive side-menu for “What-If” analysis.

## 🗂️ Project Structure
- `app1.py` - Core backend logic, demonstrating the model training lifecycle and CLI-based user parameter intake.
- `app2.py` - Full-stack Streamlit web application merging the ML model with an interactive, responsive frontend.
- `Hackathon/` - Directory holding the foundational dataset inputs: 
  - `_h_batch_process_data.xlsx` (Environment / Machine constraints)
  - `_h_batch_production_data.xlsx` (Quality / End-result metrics)
- Documentation - Accompanying PDFs and PPTX files summarizing the design concepts behind the architecture.

## 🚀 Getting Started

### Prerequisites
Make sure you have python installed. You will need `pandas`, `numpy`, `streamlit`, `scikit-learn`, `matplotlib`, and `seaborn`.

```bash
pip install -r requirements.txt
# Alternatively, install manually: pip install pandas numpy streamlit scikit-learn matplotlib seaborn openpyxl
```

### Running the CLI Engine
If you'd like to test the pure core logic from the command line:
```bash
python app1.py
```
*You will be prompted to enter sensory measurements like time, temperature, and RPM to see optimization results.*

### Running the Web Dashboard (Recommended)
To launch the interactive dashboard:
```bash
streamlit run app2.py
```

## 🧠 Optimization Scoring Formula
The AI decides what a "good" batch is by using the following mathematical weighting mechanism during operation:
```python
score = (Hardness * 0.3) + (Dissolution * 0.3) + (Uniformity * 0.2) - (Carbon Emission * 0.2)
```
This forces the system to find the perfect middle ground between high yield and low energy usage!

## 🤝 Contributing
Contributions are highly welcome. Open an issue or fork the repository to experiment with alternative `MultiOutputRegressor` architectures like Gradient Boosting!
