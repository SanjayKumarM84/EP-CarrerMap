import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings, json
import numpy as np

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
            if value in label_encoders[X.columns[i]].classes_:
                # If the label is seen, transform it
                encoded_value = label_encoders[X.columns[i]].transform([value])[0]
            else:
                # If the label is unseen, append it as a new label
                label_encoders[X.columns[i]].classes_ = np.append(label_encoders[X.columns[i]].classes_, value)
                encoded_value = len(label_encoders[X.columns[i]].classes_) - 1  # Assign a unique integer value
            user_input_encoded.append(encoded_value)
        else:
            user_input_encoded.append(value)

    # Make predictions
    predicted_job_roles_encoded = rf_model.predict([user_input_encoded])
    predicted_job_roles = label_encoders['Suggested Job Role'].inverse_transform(predicted_job_roles_encoded)

    return predicted_job_roles, None

def predictMain(inputData):
    # Example user input
    # user_input = [40,67,64,82,77,72,87,68,40,10,9,0,2,5,'no','yes','no','full stack','database security','no','yes','excellent','poor','parallel computing','system developer','job','Finance','no','yes','Childrens','salary','no','gentle','Management','work','hard worker','no','no']
    
    # inputData = json.loads(inputData)
    user_input = list(inputData.values())

    # Ensure that the input data has the correct number of features
    if len(user_input) != X_train.shape[1]:
        raise ValueError(f"Input data has incorrect number of features. Expected {X_train.shape[1]} features.")

    predicted_job_roles, _ = predict_job_roles(user_input)

    # Filter out the predicted job roles based on accuracy rate
    filtered_predicted_job_roles = [job_role for job_role in predicted_job_roles]

    print("Predicted Job Roles:", filtered_predicted_job_roles)

    return filtered_predicted_job_roles

# # Example input data
# inputData = {
#     "Acedamic percentage in Operating Systems": 69,
#     "percentage in Algorithms": 63,
#     "Percentage in Programming Concepts": 78,
#     "Percentage in Software Engineering": 87,
#     "Percentage in Computer Networks": 94,
#     "Percentage in Electronics Subjects": 94,
#     "Percentage in Computer Architecture": 87,
#     "Percentage in Mathematics": 84,
#     "Percentage in Communication skills": 61,
#     "Hours working per day": 9,
#     "Logical quotient rating": 4,
#     "hackathons": 0,
#     "coding skills rating": 4,
#     "public speaking points": 8,
#     "can work long time before system?": "yes",
#     "self-learning capability?": "yes",
#     "Extra-courses did": "yes",
#     "certifications": "shell programming",
#     "workshops": "cloud computing",
#     "talenttests taken?": "no",
#     "olympiads": "yes",
#     "reading and writing skills": "excellent",
#     "memory capability score": "excellent",
#     "Interested subjects": "cloud computing",
#     "interested career area ": "system developer",
#     "Job/Higher Studies?": "higherstudies",
#     "Type of company want to settle in?": "Web Services",
#     "Taken inputs from seniors or elders": "no",
#     "interested in games": "no",
#     "Interested Type of Books": "Prayer books",
#     "Salary Range Expected": "salary",
#     "In a Realtionship?": "no",
#     "Gentle or Tuff behaviour?": "stubborn",
#     "Management or Technical": "Management",
#     "Salary/work": "salary",
#     "hard/smart worker": "hard worker",
#     "worked in teams ever?": "yes",
#     "Introvert": "no"
# }

# # Convert the JSON object to string
# inputDataStr = json.dumps(inputData)

# # Make predictions
# predicted_job_roles = predictMain(inputData)
# print("Predicted Job Roles:", predicted_job_roles)
