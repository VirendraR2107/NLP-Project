#Note: -Mention all steps, including dependencies at each step. Users just need to change the path for relevant results.


#Importing Necessary Libraries
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from textstat import syllable_count
from collections import Counter
import re

#Download NLTK Res.
nltk.download('punkt')
nltk.download('wordnet')

#Object Creation
lem = WordNetLemmatizer()

#Importing Positive and Negative Words
with open('D:\Blackoffer\MasterDictionary/positive-words.txt', 'r', encoding='ISO-8859-1') as f:
    Pos_split = set(f.read().split())

with open('D:\Blackoffer\MasterDictionary/negative-words.txt', 'r', encoding='ISO-8859-1') as f:
    Neg_split = set(f.read().split())


#Function to Read Stopwords from File
def read_stopwords(filename):
    try:
        with open(filename, 'r', encoding='ISO-8859-1') as f:
            stopwords = f.read().split()
        return stopwords
    except FileNotFoundError:
        print(f"{filename} not found.")
        return []


#Read Stopwords
stopwords_files = ['StopWords_Auditor.txt', 'StopWords_Currencies.txt', 'StopWords_DatesandNumbers.txt',
                   'StopWords_Generic.txt', 'StopWords_GenericLong.txt', 'StopWords_Geographic.txt',
                   'StopWords_Names.txt']
all_stopwords = set()
for file in stopwords_files:
    stopwords = read_stopwords(f'D:/Blackoffer/StopWords/{file}')
    all_stopwords.update(stopwords)


#Function to Preprocess Text
def preprocess_text(text):
    string_format = str(text).lower()
    lower_words = re.sub('[^a-zA-Z]+', ' ', string_format).strip()
    lower_words = lower_words.split()
    token_word = [t for t in lower_words if t not in all_stopwords]
    lemm = [lem.lemmatize(w) for w in token_word]
    return lemm


#Function to Fetch Webpage Content and Extract Data
def fetch_data(row):
    URL_ID, URL = row['URL_ID'], row['URL']
    response = requests.get(URL)
    bs = BeautifulSoup(response.text, "html.parser")
    try:
        article_title = bs.find(name="h1").get_text()
        article_content = bs.find(attrs={"class": "td-post-content"}).get_text().strip()
        with open(f"{URL_ID}.txt", "w", encoding="utf-8") as f:
            f.write(f"Title: {article_title}\n\n")
            f.write(f"Content:\n{article_content}")
        return (URL_ID, article_title, article_content)
    except Exception as e:
        print("Error processing URL_ID:", URL_ID)
        print(e)
        return None


#Read Input Data
data = pd.read_excel(r'D:\Blackoffer\Input.xlsx')
df = data.copy()


#Fetch Article Data
Text = [fetch_data(row) for _, row in df.iterrows()]
new_df = pd.DataFrame(Text, columns=['URL_ID', 'article_title', 'article_content'])
df = pd.merge(df, new_df, on='URL_ID', how='left')
df['article'] = df.apply(lambda row: f"{row['article_title']}\n\n{row['article_content']}", axis=1)
df = df.drop(['article_title', 'article_content'], axis=1)




# Function to calculate positive score
def positive_score(text, positive_words):
    tokens = word_tokenize(text.lower())
    return sum(1 for token in tokens if token in positive_words)

# Function to calculate negative score
def negative_score(text, negative_words):
    tokens = word_tokenize(text.lower())
    return sum(1 for token in tokens if token in negative_words)

# Function to calculate polarity score
def polarity_score(pos_score, neg_score):
    return (pos_score - neg_score) / ((pos_score + neg_score) + 0.000001)

# Function to calculate subjectivity score
def subjectivity_score(pos_score, neg_score, num_words):
    return (pos_score + neg_score) / (num_words + 0.000001)


# Function to calculate average sentence length
def avg_sentence_length(text):
    sentences = sent_tokenize(text)
    return sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)

# Function to calculate percentage of complex words
def percentage_complex_words(text, complex_words):
    tokens = word_tokenize(text.lower())
    return sum(1 for token in tokens if token in complex_words) / len(tokens)

# Function to calculate Fog index
def fog_index(avg_sentence_length, percentage_complex_words):
    return 0.4 * (avg_sentence_length + percentage_complex_words)


# Function to calculate average number of words per sentence
def avg_words_per_sentence(text):
    sentences = sent_tokenize(text)
    return sum(len(word_tokenize(sentence)) for sentence in sentences) / len(sentences)

# Function to calculate number of complex words
def complex_word_count(text, complex_words):
    tokens = word_tokenize(text.lower())
    return sum(1 for token in tokens if token.lower() in complex_words)


# Function to calculate syllable per word
def syllable_per_word(text):
    tokens = word_tokenize(text)
    syllable_count_list = [syllable_count(token) for token in tokens if token.isalpha()]
    return sum(syllable_count_list) / len(syllable_count_list)

# Function to calculate number of personal pronouns
def personal_pronouns(text):
    tokens = word_tokenize(text.lower())
    personal_pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    return sum(1 for token in tokens if token in personal_pronouns)

# Function to calculate average word length
def avg_word_length(text):
    tokens = word_tokenize(text.lower())
    return sum(len(token) for token in tokens) / len(tokens)



# Preprocess text
df['complex_words'] = df['article'].apply(preprocess_text)
df['article_without_stopwords'] = df['article'].apply(lambda x: ' '.join(preprocess_text(x)))

# Text metrics
df['WORD COUNT'] = df['article'].apply(lambda x: len(word_tokenize(x)))
df['AVG SENTENCE LENGTH'] = df['article'].apply(lambda x: avg_sentence_length(x))
df['SYLLABLE PER WORD'] = df['article'].apply(lambda x: syllable_per_word(x))
df['COMPLEX WORD COUNT'] = df['article'].apply(lambda x: sum(1 for token in word_tokenize(x.lower()) if token in all_stopwords))
df['POSITIVE SCORE'] = df['article'].apply(lambda x: positive_score(x, Pos_split))
df['NEGATIVE SCORE'] = df['article'].apply(lambda x: negative_score(x, Neg_split))
df['POLARITY SCORE'] = df.apply(lambda row: polarity_score(row['POSITIVE SCORE'], row['NEGATIVE SCORE']), axis=1)
df['SUBJECTIVITY SCORE'] = df.apply(lambda row: subjectivity_score(row['POSITIVE SCORE'], row['NEGATIVE SCORE'], row['WORD COUNT']), axis=1)
df['PERCENTAGE OF COMPLEX WORDS'] = df.apply(lambda row: row['COMPLEX WORD COUNT'] / row['WORD COUNT'], axis=1)
df['FOG INDEX'] = df.apply(lambda row: fog_index(row['AVG SENTENCE LENGTH'], row['PERCENTAGE OF COMPLEX WORDS']), axis=1)
df['AVG NUMBER OF WORDS PER SENTENCE'] = df['article'].apply(avg_words_per_sentence)
df['PERSONAL PRONOUNS'] = df['article'].apply(personal_pronouns)
df['AVG WORD LENGTH'] = df['article'].apply(avg_word_length)

# Drop three columns from DataFrame
df = df.drop(['article','complex_words','article_without_stopwords'], axis=1)


# Column arrangements
desired_columns = ['URL_ID','URL','POSITIVE SCORE','NEGATIVE SCORE','POLARITY SCORE','SUBJECTIVITY SCORE','AVG SENTENCE LENGTH',
                   'PERCENTAGE OF COMPLEX WORDS','FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE',
                   'COMPLEX WORD COUNT','WORD COUNT','SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH']  



# Create a new DataFrame with columns in the desired order
df = df[desired_columns]
FINAL_DF=pd.DataFrame(df)
FINAL_DF.head()


# Save DataFrame to CSV file
FINAL_DF.to_csv('Final_Output.csv', index=False)

print(FINAL_DF.head())