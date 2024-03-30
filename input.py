import pandas as pd
import json

# Read CSV file into pandas DataFrame
df = pd.read_csv("dataset.csv")

# Convert DataFrame to JSON
json_data = df.to_json(orient="records")

# Convert JSON data to list of records
records = json.loads(json_data)

# Print only the first record
print(json.dumps(records[0], indent=4))  # This will print the first record with indentation for better readability
