
# 🎖️ Defense Candidate EDA + ML Dashboard

## 📝 Overview

This is an interactive web-based dashboard built using **Streamlit** that enables **Exploratory Data Analysis (EDA)** and **Machine Learning-based prediction** on a dataset of defense candidates. The app includes **user authentication** (login/register) and provides insights into candidate selection patterns, OLQ scores, SSB results, and includes a prediction tool to assess candidate recommendation probability.

---

## 🚀 Features

- 🔐 **User Authentication**: Secure login and registration system with password hashing.
- 📊 **EDA Dashboard**: Interactive visualizations using Plotly for analyzing trends in:
  - SSB Scores
  - OLQ Scores vs Recommendation
  - Region-wise recommendation stats
- 🤖 **ML Module**: 
  - Trains a Random Forest Classifier to predict candidate recommendation based on:
    - Age
    - OLQ Score
    - Psychological Test Score
    - GTO Result
    - PI Marks
  - Displays classification report and confusion matrix
  - Provides live prediction form for new inputs
- 📂 **Data Persistence**: Cleaned dataset is saved automatically.

---

## 📁 Folder Structure

```
project/
│
├── defense.py                    # Main Streamlit app
├── defense_candidate_dataset.csv # Input dataset (required)
├── cleaned_defense_dataset.csv   # Cleaned version saved by app
├── users.csv                    # Stores hashed user credentials
└── README.md                    # Project documentation
```

---

## ⚙️ Installation & Run

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-repo/defense-dashboard.git
   cd defense-dashboard
   ```

2. **Install Requirements**  
   *(Make sure Python 3.8+ and pip are installed)*
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Dataset**  
   Place `defense_candidate_dataset.csv` in the root folder with necessary columns.

4. **Run the App**  
   ```bash
   streamlit run defense.py
   ```

---

## 🧾 Dataset Format (Sample Columns)

The dataset (`defense_candidate_dataset.csv`) should include the following columns:

- `Age`
- `OLQ_Score`
- `Psych_Test`
- `GTO_Result`
- `PI_Marks`
- `Recommended` (Yes/No)
- `SSB_Score`
- `Region`
- `Gender`
- `Rank_Secured` (optional for visualization)

---

## 📊 ML Model Details

- **Algorithm**: Random Forest Classifier
- **Training Split**: 80/20 train-test
- **Target**: `Recommended` (binary classification)
- **Features Used**:
  - Age
  - OLQ_Score
  - Psych_Test
  - GTO_Result
  - PI_Marks

---

## 🔐 Authentication System

- New users can **register** using the sidebar.
- Existing users can **login** to access dashboard.
- Credentials stored in `users.csv` with **SHA-256 hashed passwords** for security.

---

## ✅ To-Do / Future Improvements

- Add email-based OTP verification
- Visual improvements and theming
- Support for admin analytics (e.g., total users, prediction stats)
- Export prediction logs for analysis

---

## 🙌 Contribution

Feel free to fork and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
