# Essay Summarization and Automated Grading System

## Project Overview

This project addresses the challenge of efficiently grading large volumes of essays, a task that traditionally demands significant time and manual effort from instructors. The goal is to develop a system capable of automatically summarizing essays and assigning grades, thereby streamlining the grading process, reducing human bias, and improving consistency.

## Problem Statement

Grading essays is a time-consuming and often subjective process. The lack of an efficient system for automated essay grading motivated the development of this project, which aims to provide a solution that can automatically summarize and grade essays. The summarized text also serves as a quick review option for instructors to verify the assigned grade.

## Methodology

The project is divided into several key phases:

1. **Data Loading**: Reading the essay data from files.
2. **Exploratory Data Analysis (EDA)**: Analyzing data to understand patterns and relationships.
3. **Data Preprocessing**: Cleaning and preparing the data for the model.
4. **Feature Extraction**: Converting text data into a machine learning-friendly format.
5. **Model Training**: Training machine learning models on the preprocessed data.
6. **Model Evaluation**: Evaluating model performance using various metrics.
7. **Hyperparameter Tuning**: Optimizing model parameters for better performance.
8. **Prediction and Analysis**: Using the trained model to predict and analyze new essays.

## Models Used

### NLP Model
The NLP model is responsible for text cleaning and summarization. The summarized essays are saved in an Excel sheet along with the corresponding scores.

### Machine Learning Models
The summarized essays undergo further preprocessing and are passed to machine learning models, which learn from the data and predict scores based on the summary. Three primary models were employed:

1. **Logistic Regression**: Used for binary classification tasks, this model classifies the summarized essay to predict scores.
2. **Decision Tree**: A supervised learning algorithm that classifies essays based on a series of binary decisions.
3. **Neural Network**: A deep learning model inspired by the human brain, capable of learning from observational data and making predictions based on the learned features.

## Dataset

The dataset comprises 8 distinct sets of essays, each corresponding to different prompts. These essays, written by students in grades 7 to 10, vary in length from 150 to 550 words and have been hand-graded by multiple raters to ensure reliability. The dataset includes the following columns:

- `Essay_id`: Unique identifier for each student essay.
- `essay_set`: Numerical identifier indicating the essay set.
- `essay`: Text content of the student's response.
- `rater1_domain1` to `rater3_domain1`: Scores from different raters for the primary domain.
- `domain1_score`: Final agreed-upon score after resolving differences.
- Additional columns for secondary domains and trait scores for certain essay sets.

## Data Preprocessing and Visualization

- **Normalization**: The target scores were normalized due to varying scales in different columns.
- **Data Cleaning**: Regular expressions were used to clean HTML tags, special symbols, punctuation, numbers, and whitespaces, converting text to lowercase.
- **Text Summarization**: Summarized essays were generated using the spaCy library and saved for model training.
- **Handling Null Values**: Any null fields were cleaned before training the models.

## Implementation Details

### Libraries Used

- **spaCy**: For advanced NLP tasks.
- **Regular Expressions (re)**: For complex text pattern matching and cleaning.
- **NumPy**: For numerical computations.
- **Pandas**: For data processing and analysis.
- **Seaborn & Matplotlib**: For data visualization.
- **scikit-learn**: For model training and evaluation.

### Model Training and Evaluation

- **Logistic Regression**: Utilized `TfidfVectorizer` to convert text to numerical features. The model was trained on 80% of the data and evaluated using metrics like accuracy, precision, recall, F1 score, and confusion matrix.
- **Decision Tree**: Similar to Logistic Regression, trained and evaluated with appropriate metrics.
- **Neural Network**: Configured with dense layers and dropout layers to prevent overfitting. Hyperparameters were tuned using Keras Tuner with Random Search.

## Contributors

- **Sai Gottumukkala**: Data visualization and preprocessing.
- **Sai Charan Merugu**: Neural network model design.
- **Swetha Guntupalli**: Logistic regression model design.
- **Karishma Bollineni**: Decision tree model design and implementation.

## Challenges Faced

- Limited data availability, leading to loss of data during cleaning and summarization, which impacted the performance of the machine learning models.

## References

- The Hewlett Foundation: Automated Essay Scoring [Link](https://www.kaggle.com/c/asap-aes/data)
- Text Summarization [Link](https://cs.nyu.edu/~kcho/DMQA/)
- Keras Tuner Documentation [Link](https://keras.io/keras_tuner/)

