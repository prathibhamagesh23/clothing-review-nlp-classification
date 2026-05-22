#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Prathibha Magesh
# #### Student ID: s3859590
# 
# 
# Environment: Python 3 and Jupyter notebook
# 
# ## Libraries Used
# - **pandas** for handling the dataset and saving outputs
# - **re** (regular expressions) for extracting tokens from text
# - **collections.Counter** for calculating term and document frequencies
# 
# 
# # Introduction
# 
# ## Dataset Description
# Work with a dataset of women's clothing reviews originally sourced from Kaggle (https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews). Dataset has been modified, not in original state.
# We will focus on the `Review Text` column, which contains customer feedback. Clean and preprocess this text so it can be used for classification in future tasks.
# 
# ## Task 1: Basic Text Pre-processing of Clothing Reviews
# 
# In this task, I'll clean and process the "Review Text" column from a clothing review dataset and prepares customer clothing reviews for use in machine learning models.
# This is a vital step in Natural Language Processing (NLP) because text data is often messy and unstructured, so it's essential to clean and format it properly before analysis. This preprocessing step converts raw reviews into a structured format suitable for feature extraction.
# 
# ### Steps involved:
# - Tokenize text using a custom regular expression.
# - Convert text to lowercase.
# - Remove short words (length < 2).
# - Remove stopwords from a given list.
# - Remove words that appear only once in the dataset.
# - Remove the top 20 most frequent words (based on document frequency).
# - Save the final cleaned text and vocabulary for later feature extraction.
# 

# ## Importing libraries 

# In[1]:


import pandas as pd
import re
from collections import Counter


# ### 1.1 Loading data

# In[2]:


# Load the dataset
df = pd.read_csv("assignment3.csv")
df.head() #first few rows


# In[3]:


# Load stopwords
with open("stopwords_en.txt", "r") as f:
    stopwords = set(f.read().splitlines())
print(f"Total stopwords loaded: {len(stopwords)}")


# ### 1.2 Pre-processing data
# 

# ### Step 1: Tokenization & Clean Reviews
# 
# Process each review using a regular expression that extracts words while allowing for hyphens and apostrophes (e.g., “don’t”, “t-shirt”).
# - Convert all tokens to lowercase to maintain uniformity.
# - Remove tokens shorter than two characters.
# - Exclude any word in the stopword list from the final result.
# 
# This step produces a clean list of meaningful words for each review.

# In[4]:


token_pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?" # Regular expression for tokenizing
def clean_review(text):
    if pd.isnull(text):
        return []
    tokens = re.findall(token_pattern, text) # Tokenize using regex
    cleaned = [
        word.lower()
        for word in tokens
        if len(word) >= 2 and word.lower() not in stopwords  # Convert to lowercase and filter
    ]
    return cleaned

# Apply cleaning
df['cleaned_tokens'] = df['Review Text'].apply(clean_review)

# Display an example
df[['Review Text', 'cleaned_tokens']].head()


# ## Step 2: Remove Rare and Common Words
# - Remove rare words that appear only once in the entire corpus. These may be typos or irrelevant.
# - Remove the top 20 most frequent words (how many different reviews they appear in). These are often too generic to be helpful for classification.
# 
# Combine both filters to retain only useful, informative terms.
# 

# In[5]:


# Flatten tokens into a single list
all_tokens = [word for tokens in df['cleaned_tokens'] for word in tokens]
word_freq = Counter(all_tokens) # Count frequency of each word
print("Total unique words before filtering:", len(word_freq))

# Identify rare words
rare_words = {word for word, freq in word_freq.items() if freq == 1}

# Identify top 20 most frequent word
# First calculate document frequency
doc_freq = Counter()
for tokens in df['cleaned_tokens']:
    unique_words = set(tokens)
    doc_freq.update(unique_words)
top_20_common_words = {word for word, _ in doc_freq.most_common(20)}
words_to_remove = rare_words.union(top_20_common_words) # Combine both sets for removal

# Filter the cleaned tokens
df['final_tokens'] = df['cleaned_tokens'].apply(lambda tokens: [w for w in tokens if w not in words_to_remove])
# Show results
df[['Review Text', 'final_tokens']].head()


# ## Step 3: Save Cleaned Reviews
# Convert each cleaned list of tokens back into a string with space-separated words. Save the results to a CSV file named `processed.csv`, which will be used for feature extraction in the next task.

# In[8]:


# Convert tokens to space-separated strings
df['processed_text'] = df['final_tokens'].apply(lambda tokens: ' '.join(tokens))
df[['processed_text']].to_csv("processed.csv", index=False) # Save only the processed text to CSV
print("Saved processed.csv")


# ## Saving required outputs
# Save the requested information as per specification.
# - vocab.txt

# ## Step 4: Generate Vocabulary
# Create a vocabulary from the cleaned data:
# - Collect all unique words and sort them alphabetically.
# - Assign each word a unique index starting from 0.
# - Save the vocabulary to `vocab.txt` using the format `word:index`.
# 
# This vocabulary is essential for building numerical feature representations in the next task.
# 

# In[10]:


# Build vocabulary set from all final tokens
vocab_set = set(word for tokens in df['final_tokens'] for word in tokens)
vocab_list = sorted(vocab_set) # Sort alphabetically
# Create word:index mapping
vocab_dict = {word: idx for idx, word in enumerate(vocab_list)}

# Save to vocab.txt
with open("vocab.txt", "w") as f:
    for word, idx in vocab_dict.items():
        f.write(f"{word}:{idx}\n")
print(f" Saved vocab.txt with {len(vocab_dict)} words")


# ## Summary
# - Preprocessed customer reviews using tokenization, stopword removal, and frequency-based filtering.
# - Saved the cleaned reviews in `processed.csv`.
# - Created a vocabulary index in `vocab.txt`.
# 
# These outputs complete Task 1 and serve as inputs for Task 2.
# 
