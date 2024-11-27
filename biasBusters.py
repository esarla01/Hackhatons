from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import time
import numpy as np

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
You are a dataset generation system focused on providing sentences on a given 
topic. Your responses should be concise and directly show examples of these 
expressions without preambles or qualifiers. Use conversational language and 
dialogue that represents common stereotypes in everyday speech.

Examples of non-stereotypical expressions:
- 'Yemen is located on the Arabian Peninsula. It has an ancient culture.'
- 'My mechanic is a female.'
- 'The swimmer was very fat.'
- 'The psychologist saw the client. She falls in love with her client and 
fantasizes about leaving her husband for them.'
- 'Some people claim Mohammed's message was angelic in origin.'

Ensure all generated sentences align with these examples, avoiding any language 
that generalizes or reinforces stereotypes while focusing on factual, neutral 
descriptions.
"""

def get_openai_response(question):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":content1},
            {"role":"user", "content": question}
        ],
        max_tokens=200
    )
    return completion.choices[0].message.content

def get_openai_response2(question):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":content2},
            {"role":"user", "content": question}
        ],
        max_tokens=200
    )
    return completion.choices[0].message.content

def create_sentences(domain, topics, word_range, num_sentences):
    """
    Generate stereotypical and unstereotypical sentences based on the given domain.
    """
    sentences = []
    labels = []

    for _ in tqdm(range(num_sentences)):
        topic = np.random.choice(topics)
        if domain != 'unrelated':
            # Generate stereotypical sentence
            stereotypical_prompt = (
                f"Generate a sentences exhibiting stereotype about {topic}. The sentence should be between {word_range[0]} and {word_range[1]} words."
            )
            stereotypical_sentence = get_openai_response(stereotypical_prompt)
            sentences.append(stereotypical_sentence)
            labels.append(f"stereotype_{domain}")

        # Generate unstereotypical sentence
        unstereotypical_prompt = (
            f"Generate a sentence about {topic}. The sentence should be between {word_range[0]} and {word_range[1]} words."
        )
        unstereotypical_sentence = get_openai_response2(unstereotypical_prompt)
        sentences.append(unstereotypical_sentence)
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
        temp_sent, temp_label = create_sentences(stereotype, st_dict[stereotype], (5, 30), 200)
        sentences.extend(temp_sent)
        labels.extend(temp_label)

    # Save sentences to pandas DataFrame
    df = save_to_pandas(sentences, labels)
    df.to_csv("stereotypes.csv")