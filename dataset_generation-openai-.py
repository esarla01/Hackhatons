import threading
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import time
import re
import tiktoken


load_dotenv()
client = OpenAI()

content1 = """ 
You are a dataset generation system focused on providing sentences that exhibit stereotypical expressions. Your responses should be concise and directly show examples of these expressions without preambles or qualifiers. Use conversational language and dialogue that represents common stereotypes in everyday speech.

Examples of stereotypes:
- 'Saudi Arabian culture is in reality highly extremist.'
- 'Asexual people are out to push an agenda on society/children/etc.'
- 'Dutch people live in the Netherlands. They are all tall and white.'
- 'Ukrainian people are wonderful people but are constantly fighting for their rights.'

Always focus on generating sentences that explicitly reflect stereotypes, avoiding factual descriptions or subjective storytelling.
"""

content2 = """
You are a dataset generation system focused on providing sentences on a given topic. Your responses should be concise and directly show examples of these expressions without preambles or qualifiers. Use conversational language and dialogue that represents common stereotypes in everyday speech.

Examples of non-stereotypical expressions:
- 'Yemen is located on the Arabian Peninsula. It has an ancient culture.'
- 'My mechanic is a female.'
- 'The swimmer was very fat.'
- 'The psychologist saw the client. She falls in love with her client and fantasizes about leaving her husband for them.'
- 'Some people claim Mohammed's message was angelic in origin.'

Ensure all generated sentences align with these examples, avoiding any language that generalizes or reinforces stereotypes while focusing on factual, neutral descriptions.
"""

history1 = [content1]
history2 = [content2]

def split_into_sentences(text):
    sentence_endings = r'&'
    
    sentences = re.split(sentence_endings, text.strip())
    
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def get_openai_response(question):
    global history1
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content": content1},
            {"role":"user", "content": question}
        ],
        max_tokens=4096
    )
    response = completion.choices[0].message.content
    #history1.append({"role":"user", "content": question})
    history1.append(response)
    return response

def get_openai_response2(question):
    global history2
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":content2},
            {"role":"user", "content": question}
        ],
        max_tokens=4096 
    )
    response = completion.choices[0].message.content
    #history2.append({"role":"user", "content": question})
    history2.append(response)
    return response


def create_sentences(domain, topics,  word_range, num_sentences):
    """
    Generate stereotypical and unstereotypical sentences based on the given domain.
    """
    sentences = []
    labels = []

    topic = np.random.choice(topics)
    if domain != 'unrelated':

        stereotypical_prompt = (
             f"Generate exactly {num_sentences} distinct sentences exhibiting stereotype about {topic}. The sentence should be between {word_range[0]} and {word_range[1]} words."
        )
        stereotypical_sentences = get_openai_response(stereotypical_prompt)
        stereotypical_sentences = split_into_sentences(stereotypical_sentences)
        for sentence in stereotypical_sentences:
            sentences.append(sentence)
            labels.append(f"stereotype_{domain}")

    # Generate unstereotypical sentence
    unstereotypical_prompt = (
        f"Generate exactly {num_sentences} distinct sentence about {topic}. The sentence should be between {word_range[0]} and {word_range[1]} words."
    )
    unstereotypical_sentences = get_openai_response2(unstereotypical_prompt)
    unstereotypical_sentences = split_into_sentences(unstereotypical_sentences)
    for sentence in unstereotypical_sentences:
        sentences.append(sentence)
        if domain =='unrelated':
            labels.append('unrelated')
        else:
            labels.append(f"neutral_{domain}")

    return sentences, labels

def save_to_pandas(sentences, labels):
    """
    Save sentences and labels to a pandas DataFrame.
    """
    df = pd.DataFrame({"Sentence": sentences, "Label": labels})
    print(df)  # Print the DataFrame to the console
    return df

# Function to count tokens
def count_tokens(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0

    for message in messages:
        # Add tokens for role, content, and separator (3 tokens per message)
        total_tokens += 4 + len(encoding.encode(message))

    return total_tokens
# Estimate API energy consumption
def estimate_api_request_energy(requests_count, energy_per_request=0.01):
    return requests_count * energy_per_request  # kWh

# Include energy for the API call in the carbon footprint calculation
def calculate_carbon_footprint_with_api(messages_list, num_api_requests, model="gpt-3.5-turbo", emissions_factor=0.43, energy_per_token=0.0003, energy_per_request=0.01):
    total_tokens = 0
    print(messages_list)
    for messages in messages_list:
        print('messages')
        total_tokens += count_tokens(messages, model=model)
    
    # Total energy from tokens and API requests
    energy_from_tokens = total_tokens * energy_per_token  # kWh
    energy_from_api = estimate_api_request_energy(num_api_requests, energy_per_request)  # kWh
    total_energy = energy_from_tokens + energy_from_api

    # Calculate carbon footprint
    carbon_footprint = total_energy * emissions_factor  # kg CO₂e
    return total_tokens, energy_from_tokens, energy_from_api, total_energy, carbon_footprint


if __name__ == "__main__":
    # Get user inputs
    # domain = input("Enter the domain (e.g., athletes, scientists, etc.): ")
    # min_words = int(input("Enter the minimum word count for a sentence: "))
    # max_words = int(input("Enter the maximum word count for a sentence: "))
    # num_sentences = int(input("Enter the number of sentences to generate for each category: "))
    st_dict ={
        'unrelated': ['Parks', 'Cities', 'History', 'Sports', 'Science', 'Education', 'Technology', 'Space', 'Nature'],
        'nationality': ['italian', 'turkish', 'polish', 'british', 'australian', 'american', 'german', 'swiss', 'mexican', 'asian', 'african'],
        'profession': ['lawyer', 'doctor', 'janitor', 'chef', 'banker', 'barista', 'bartender', 'clerk', 'driver'],
        'gender': ['Men', 'Women'],
        'lgbtq+': ['Gay', 'Lesbian', 'Transgender', 'Gender Neutral', 'Queer', 'Furry'],
        'religion': ['Christian', 'Muslim', 'Buddist', 'Jewish', 'Atheist', 'Hindu'],
        'race': ['Black People', 'White People', 'Asian People', 'Middle Eastern People', 'African People', 'Brown People', 'Latino People']
    }

    sentences, labels = [], []
    
    # Generate sentences
    for stereotype in tqdm(st_dict, desc="Processing..."):
        temp_sent, temp_label = create_sentences(stereotype,st_dict[stereotype], (5, 30), 3)
        sentences.extend(temp_sent)
        labels.extend(temp_label)

          
    # Save sentences to pandas DataFrame
    df = save_to_pandas(sentences, labels)
    df.to_csv("stereotypes3.csv")
