import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings

# Load the dataset
df = pd.read_csv("dataset.csv")

# Separate features and target variable
X = df.drop('Suggested Job Role', axis=1)
y = df['Suggested Job Role']

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include='object').columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Encode target variable 'Suggested Job Role'
label_encoders['Suggested Job Role'] = LabelEncoder()
y = label_encoders['Suggested Job Role'].fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_model.predict(X_test)

# Calculate accuracy rate
accuracy = accuracy_score(y_test, y_pred) * 100

# Function to preprocess user input and make predictions
def predict_job_roles(user_input):
    # Preprocess user input
    user_input_encoded = []
    for i, value in enumerate(user_input):
        if isinstance(value, str):
            user_input_encoded.append(label_encoders[X.columns[i]].transform([value])[0])
        else:
            user_input_encoded.append(value)

    # Make predictions
    predicted_job_roles_encoded = rf_model.predict([user_input_encoded])
    predicted_job_roles = label_encoders['Suggested Job Role'].inverse_transform(predicted_job_roles_encoded)

    return predicted_job_roles, None

# Example user input
user_input = [40,67,64,82,77,72,87,68,40,10,9,0,2,5,'no','yes','no','full stack','database security','no','yes','excellent','poor','parallel computing','system developer','job','Finance','no','yes','Childrens','salary','no','gentle','Management','work','hard worker','no','no']
predicted_job_roles, _ = predict_job_roles(user_input)

# Filter out the predicted job roles based on accuracy rate
filtered_predicted_job_roles = [job_role for job_role in predicted_job_roles]

print("Predicted Job Roles", filtered_predicted_job_roles)
