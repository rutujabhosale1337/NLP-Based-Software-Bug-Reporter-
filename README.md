NLP-Based Software Bug Reporter

An NLP-based application that analyzes software bug descriptions and generates structured bug reports by predicting bug types and severity levels using Machine Learning.

Project Overview

Software bug reports often contain unstructured text, making it difficult for developers and testers to classify issues and prioritize fixes efficiently.

The NLP-Based Software Bug Reporter aims to simplify this process by processing textual bug descriptions and predicting relevant bug categories and severity levels. It combines Natural Language Processing (NLP) and Machine Learning to support more organized and efficient bug reporting.

Objectives

Analyze software bug descriptions using NLP techniques.

Classify bugs into relevant categories.

Predict the severity of reported bugs.

Generate structured information to support bug tracking and prioritization.

Explore the use of Machine Learning in software testing and maintenance.

Key Features

Bug Type Classification: Predicts the category of a software bug.

Severity Prediction: Estimates the severity level of a reported bug.

Text Processing: Uses NLP techniques to process textual bug descriptions.

Machine Learning Models: Uses trained models to generate predictions.

Interactive Interface: Provides a user interface for submitting bug descriptions and viewing results.

Tech Stack

Python, NLP, NLTK, Machine Learning, Scikit-learn, Data Processing, Pandas, NumPy, Model Storage, Joblib / Pickle-based model files
, Application Interface, Streamlit, Development Tools, VS Code, Git, GitHub

Project Structure
<img width="1110" height="1172" alt="image" src="https://github.com/user-attachments/assets/9c8096ac-80bd-464a-ba4b-063164054474" />


NLP-Based-Software-Bug-Reporter/
│
├── app.py                  # Application interface
├── train_model.py          # Model training script
├── bugs.csv                # Bug dataset
├── reports.csv             # Report data
├── bug_type_model.pkl      # Trained bug type model
├── severity_model.pkl      # Trained severity model
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # Project documentation


Model Performance
The models achieved the following accuracy results during the reported evaluation:

Prediction Task

Accuracy

Bug Type Classification
70%

Severity Prediction
80%

These results represent the reported model performance and may vary depending on the dataset, preprocessing, and evaluation method.

Installation and Setup
Prerequisites
Python installed on your system
Git
pip package manager

1. Clone the Repository

git clone https://github.com/rutujabhosale1337/NLP-Based-Software-Bug-Reporter-.git

2. Navigate to the Project Directory

cd NLP-Based-Software-Bug-Reporter-

3. Create a Virtual Environment (Recommended)

python -m venv venv

Activate it on Windows PowerShell:

.\venv\Scripts\Activate.ps1

4. Install Dependencies

python -m pip install -r requirements.txt

If a required dependency is missing, install it only after checking the imports used by the project.

5. Run the Application

streamlit run app.py

Open the local URL displayed in your terminal, usually:

http://localhost:8501

Model Training

To train the machine learning models, run:

python train_model.py

Run the training script before launching the application if the trained model files are missing. Ensure the dataset and required columns are available.
Result:-
<img width="975" height="368" alt="image" src="https://github.com/user-attachments/assets/c9635fd6-9e8a-4c8d-973d-97f762a27147" />
<img width="975" height="332" alt="image" src="https://github.com/user-attachments/assets/d35f0b74-a12a-4325-aa50-a7a137638130" />
<img width="975" height="427" alt="image" src="https://github.com/user-attachments/assets/2a88da69-ebc7-4c0d-aa7f-9eaedac70741" />
<img width="975" height="357" alt="image" src="https://github.com/user-attachments/assets/657f6cda-560e-402c-940c-00a2338ba304" />
<img width="975" height="346" alt="image" src="https://github.com/user-attachments/assets/07b1b95c-62bd-4b37-9b9e-1497598cac5d" />
<img width="975" height="427" alt="image" src="https://github.com/user-attachments/assets/49fed69b-a7fe-4405-a709-f8ea5fadd624" />

How It Works

Input: The user provides a software bug description.

Text Processing: The description is processed for machine learning.

Bug Classification: The trained model predicts the bug category.

Severity Prediction: Another trained model estimates the bug's severity.

Output: The application displays the prediction results to help organize the bug report.

Learning Outcomes

Through this project, I explored:

Natural Language Processing for text-based applications.

Machine Learning model training and evaluation.

Text preprocessing and data handling.

Integrating trained models into an application interface.

Building and documenting a Python-based AI/ML project.

Future Enhancements

Improve model accuracy through dataset expansion and preprocessing.

Add confidence scores to predictions.

Support more bug categories and severity levels.

Integrate with issue-tracking platforms such as GitHub Issues.

Explore transformer-based NLP models for improved text understanding.

Add export functionality for structured bug reports.

Limitations

Predictions depend on the quality and coverage of the training dataset.

The model may misclassify descriptions that are ambiguous or outside its training distribution.

Predicted severity should be reviewed by a developer or tester before prioritizing critical issues.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Author

Rutuja Sunil Bhosale

GitHub: [@rutujabhosale1337](https://github.com/rutujabhosale1337)

LinkedIn: [Rutuja Bhosale](https://www.linkedin.com/in/rutuja-bhosale-26b3a5376 )

If you find this project useful, consider starring the repository!
