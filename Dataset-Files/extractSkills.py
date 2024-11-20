import pandas as pd
import json
from bs4 import BeautifulSoup

# Function to extract Technical Skills from the JSON data
def extract_technical_skills(json_data):
    try:
        parsed_data = json.loads(json_data)
        for section in parsed_data['sections']:
            if section['title'] == 'Technical Skills':
                return section['description']
    except (json.JSONDecodeError, KeyError) as e:
        return f"Error parsing JSON: {e}"
    return "Technical Skills not found."

# Function to convert HTML to plain text
def html_to_text(html_content):
    return BeautifulSoup(html_content, 'html.parser').get_text()

# Read the CSV file
csv_file_path = '../Resources/ResourceData.csv'  # Change this to your CSV file path
df = pd.read_csv(csv_file_path)

# Assuming the column with JSON data is named 'json_column_name'
json_column_name = 'cv_section'  # Change this to the actual column name

# Display Technical Skills from each row as plain text
for index, row in df.iterrows():
    technical_skills_html = extract_technical_skills(row[json_column_name])
    technical_skills_text = html_to_text(technical_skills_html)
    print(f"Row {index + 1} - {row[""]} - Technical Skills:\n{technical_skills_text.strip()}\n")
