import openai
from openai import OpenAI
import json
import os 
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Set your OpenAI API key here
openai.api_key = api_key

# Function to convert the extended dataset into a format suitable for uploading
def prepare_and_upload_data(data):
    # Convert the prompt-response pairs into a list of dictionaries
    formatted_data = [{"prompt": pr[0], "completion": pr[1]} for pr in data]
    
    # Write the data to a temporary JSONL file (JSON Lines format)
    file_path = 'fine_tuning_data.jsonl'
    with open(file_path, 'w') as f:
        for item in formatted_data:
            f.write(json.dumps(item) + '\n')
    
    # Upload the file to OpenAI
    response = openai.File.create(file=open(file_path), purpose='fine-tune')
    print(f"Uploaded file ID: {response.id}")
    return response.id

file_path = "extended_fine_tuning_data.jsonl"
# Prepare and upload the data
file_id = prepare_and_upload_data(file_path)

# Initiate the fine-tuning process
fine_tune_response = openai.FineTune.create(
    training_file=file_id,
    model='gpt-3.5-turbo',
)
print(f"Fine-tuning job ID: {fine_tune_response.id}")
