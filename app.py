from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load your data and preprocess it as before
df = pd.read_csv('Salary Prediction of Data Professions.csv')

# Perform feature engineering and preprocessing as in your script
df['TENURE'] = (pd.to_datetime(df['CURRENT DATE']) - pd.to_datetime(df['DOJ'])).dt.days
df['AGE_PERFORMANCE'] = df['AGE'] * df['RATINGS']
df.drop(['FIRST NAME', 'LAST NAME', 'DOJ', 'CURRENT DATE'], axis=1, inplace=True)
missing_values = df.isnull().sum()

# Create new features or transform existing ones


# Drop unnecessary columns after feature engineering


# Fill missing values with mean for numerical columns
df['AGE'].fillna(df['AGE'].mean(), inplace=True)
df['RATINGS'].fillna(df['RATINGS'].mean(), inplace=True)
df['LEAVES USED'].fillna(df['LEAVES USED'].mean(), inplace=True)
df['LEAVES REMAINING'].fillna(df['LEAVES REMAINING'].mean(), inplace=True)
df['TENURE'].fillna(df['TENURE'].mean(), inplace=True)
df['AGE_PERFORMANCE'].fillna(df['AGE_PERFORMANCE'].mean(), inplace=True)
print('Missing Values:\n', missing_values)

# Fill missing categorical values with mode (most frequent value)
df['SEX'].fillna(df['SEX'].mode()[0], inplace=True)
X = df.drop('SALARY', axis=1)
y = df['SALARY']
# Define the preprocessor and model as before
numeric_features = ['AGE', 'RATINGS', 'LEAVES USED', 'LEAVES REMAINING', 'PAST EXP', 'TENURE', 'AGE_PERFORMANCE']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['SEX', 'DESIGNATION']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X = df.drop('SALARY', axis=1)
y = df['SALARY']

X_processed = preprocessor.fit_transform(X)

model = LinearRegression()
model.fit(X_processed, y)

# Define the route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs from the form
    age = int(request.form['age'])
    sex = request.form['sex']
    ratings = float(request.form['ratings'])
    leaves_used = int(request.form['leaves_used'])
    leaves_remaining = int(request.form['leaves_remaining'])
    past_exp = float(request.form['past_exp'])
    tenure = float(request.form['tenure'])
    age_performance = float(request.form['age_performance'])
    designation = request.form['designation']

    # Prepare input data for prediction
    new_data = pd.DataFrame({
        'AGE': [age],
        'SEX': [sex],
        'RATINGS': [ratings],
        'LEAVES USED': [leaves_used],
        'LEAVES REMAINING': [leaves_remaining],
        'PAST EXP': [past_exp],
        'TENURE': [tenure],
        'AGE_PERFORMANCE': [age_performance],
        'DESIGNATION': [designation]
    })

    # Preprocess the new data using the preprocessor
    new_data_processed = preprocessor.transform(new_data)

    # Make prediction
    predicted_salary = model.predict(new_data_processed)[0]

    return render_template('index.html', prediction_text=f'Predicted Salary: ₹{predicted_salary:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
