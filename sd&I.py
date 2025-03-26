import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time
import streamlit as st

# Azure OpenAI API settings
openai_api_key = ""  # Directly including your API key
openai_api_base = ""
api_version_translation = ""
translation_deployment_name = ""
embedding_deployment_name = ""

import openai
# Set up OpenAI API configuration
openai.api_key = ""
openai.api_base = ""
api_version_translation = ""
translation_deployment_name = ""

# Translation function with error handling
def translate_text(text, retries=3):
    if not text or pd.isna(text):
        return None  # Skip empty or invalid text
    
    # Set up headers for the request
    headers = {
        "Content-Type": "application/json",
        "api-key": openai.api_key,  # Use the openai.api_key for authentication
    }
    
    # Construct the URL for the Azure deployment
    url = f"{openai.api_base}/openai/deployments/{translation_deployment_name}/chat/completions?api-version={api_version_translation}"
    
    # Prepare the data for the request
    data = {
        "messages": [
            {"role": "system", "content": "Translate the following text into English."},
            {"role": "user", "content": text},
        ],
        "max_tokens": 500,
    }
    
    # Retry logic
    for attempt in range(retries):
        try:
            # Send the POST request
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # This will raise an exception for bad responses (e.g., 4xx, 5xx)
            
            # Parse the JSON response
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()  # Return the translated text
            
            return None  # Return None if there's no valid content in the response

        except requests.exceptions.HTTPError as e:
            # Retry on HTTP 429 (Too Many Requests)
            if e.response.status_code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                st.warning(f"HTTP error during translation: {e}")
                return None

        except Exception as e:
            # Catch any other unexpected errors
            st.warning(f"Unexpected error during translation: {e}")
            return None

# Function to fetch embedding with error handling
def get_embedding(text, model=embedding_deployment_name, retries=3):
    if not text or pd.isna(text):
        return None
    headers = {
        "Content-Type": "application/json",
        "api-key": openai_api_key,
    }
    url = f"{openai_api_base}/openai/deployments/{model}/embeddings?api-version={api_version_translation}"
    data = {"input": text}
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            embedding = response.json().get('data', [{}])[0].get('embedding', None)
            if embedding:
                return embedding
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                st.warning(f"Error fetching embedding: {e}")
                return None

        except Exception as e:
            st.warning(f"Unexpected error during embedding fetch: {e}")
            return None

# Function to generate embeddings for a list of records
def generate_embeddings(records, batch_size=16):
    embeddings = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        batch_embeddings = []
        for record in batch:
            embedding = get_embedding(record)
            if embedding is not None:
                batch_embeddings.append(embedding)
            else:
                st.warning(f"Skipping record due to embedding fetch failure: {record}")
        embeddings.extend(batch_embeddings)
    return embeddings

# Streamlit app setup
st.set_page_config(page_title="SD&I", layout="wide")
st.title("SD&I Translation and Recommendation")
st.write("Upload an Excel file with at least two sheets to get started.")

# File uploader
file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if file:
    try:
        # Read the sheet names
        sheet_names = pd.ExcelFile(file).sheet_names
        
        if len(sheet_names) < 2:
            st.error("The uploaded file must have at least two sheets.")
        else:
            sheet1_name = st.selectbox("Select the first sheet (Segment and Explanation)", sheet_names)
            sheet2_name = st.selectbox("Select the second sheet (Description)", sheet_names)
            
            df1 = pd.read_excel(file, sheet_name=sheet1_name)
            df2 = pd.read_excel(file, sheet_name=sheet2_name)

            if 'Segment' not in df1.columns or 'Explanation' not in df1.columns:
                st.error("First sheet must have 'Segment' and 'Explanation' columns.")
            elif 'Description' not in df2.columns:
                st.error("Second sheet must have 'Description' column.")
            else:
                explanations = df1['Explanation']
                segments = df1['Segment']
                descriptions = df2['Description']

                st.write("Translating descriptions...")
                translated_descriptions = [translate_text(desc) for desc in descriptions]

                st.write("Generating embeddings...")
                embeddings_explanations = generate_embeddings(explanations.tolist())
                embeddings_descriptions = generate_embeddings(translated_descriptions)

                similarity_results = []
                for i, embedding_desc in enumerate(embeddings_descriptions):
                    if embedding_desc is not None:
                        similarities = cosine_similarity([embedding_desc], embeddings_explanations)[0]
                        top_index = np.argmax(similarities)
                        similarity_results.append({
                            "Description": descriptions[i],
                            "Translated Description": translated_descriptions[i],
                            "Segment Classified As": segments[top_index],
                            "Highest Similarity Score": similarities[top_index]
                        })

                results_df = pd.DataFrame(similarity_results)
                st.write("Results:")
                st.dataframe(results_df)

                csv = results_df.to_csv(index=False)
                st.download_button("Download Results", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error processing the file: {e}")







