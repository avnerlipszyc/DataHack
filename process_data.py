import numpy as np
from transformers import pipeline
import language_tool_python
from tqdm import tqdm
import pandas as pd

COLUMNS = {"reviewerID": 0,
           "asin": 1,
           "reviewerName": 2,
           "helpful": 3,
           "reviewText":4,
           "overall":5,
           "summary":6,
           "unixReviewTime":7,
           "reviewTime":8,
           "llm":9}

def main():
    raw_df = pd.read_csv("./Amazon_reviews_plus_LLM.csv")
    processed_df = raw_df.copy()

    # remove columns where llm is true
    processed_df = processed_df[processed_df["llm"] == False]

    processed_df = processed_df[["reviewerID", "asin"]]

    
    # process data
    for index, row in tqdm(raw_df.iterrows(), total=raw_df.shape[0]):
        processed_df.loc[index, "helpful_percentage"] = get_helpful_percentage(row)
        processed_df.loc[index, "helpful_total"] = get_helpful_total(row)
        processed_df.loc[index, "rating"] = get_rating(row)
        processed_df.loc[index, "length_of_review"] = get_length_of_review(row)
        # processed_df.loc[index, "grammar_errors"] = get_grammar_errors(row)
        # processed_df.loc[index, "sentiment_analysis"] = get_sentiment_analysis(row[COLUMNS["reviewText"]])
        processed_df.loc[index, "avg_word_length"] = avg_word_length(row[COLUMNS["reviewText"]])
        processed_df.loc[index, "avg_sentence_length"] = avg_sentence_length(row[COLUMNS["reviewText"]])
        processed_df.loc[index, "avg_num_sentences"] = avg_num_sentences(row[COLUMNS["reviewText"]])
        processed_df.loc[index, "percent_capitalized"] = percent_capitalized(row[COLUMNS["reviewText"]])
        processed_df.loc[index, "percent_numerals"] = percent_numerals(row[COLUMNS["reviewText"]])
    
    processed_df.to_csv("./processed_reviews_without_sentiment.csv", index=False)

def get_helpful_percentage(row):
    voting_data = row[COLUMNS["helpful"]].replace("[", "").replace("]", "").replace(" ", "").split(",")
    upvotes = int(voting_data[0])
    total_votes = int(voting_data[1])

    if total_votes > 0:
        return upvotes / total_votes
    return 0.0

def get_helpful_total(row):
    voting_data = row[COLUMNS["helpful"]].replace("[", "").replace("]", "").replace(" ", "").split(",")
    total_votes = int(voting_data[1])

    return total_votes

def get_length_of_review(row):
    if isinstance(row[COLUMNS["reviewText"]], str):
        return len(row[COLUMNS["reviewText"]])
    return 0

def get_grammar_errors(row):
    is_bad_rule = lambda rule: rule.message == 'Possible spelling mistake found.' and len(rule.replacements) and rule.replacements[0][0].isupper()
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(row[COLUMNS["reviewText"]])
    matches = [rule for rule in matches if not is_bad_rule(rule)]
    return len(matches)

def get_rating(row):
    return row[COLUMNS["overall"]]/5


# negative is 1 and positive is 0
def get_sentiment_analysis(text):
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_model(text)[0]
    return result["score"] if result["label"] == "POSITIVE" else 1-result["score"]

def percent_numerals(text):
    if isinstance(text, str):
        total_chars = len(text)
        numeral_chars = sum(c.isdigit() for c in text)
        return (numeral_chars / total_chars) * 100  
    return 0.0

def avg_word_length(text):
    if not isinstance(text, str):
        return 0.0
    words = str(text).split()  
    word_lengths = [len(word) for word in words]  
    return np.mean(word_lengths) 

def avg_sentence_length(text):
    if not isinstance(text, str):
        return 0.0
    sentences = str(text).split('.')
    sentence_lengths = [len(sentence.split()) for sentence in sentences]  
    return np.mean(sentence_lengths)

def avg_num_sentences(text):
    if not isinstance(text, str):
        return 0.0
    sentences = str(text).split('.')  
    num_sentences = len(sentences)  
    return num_sentences 

def percent_capitalized(text):
  if not isinstance(text, str):
    return 0.0
  total_letters = sum(c.isalpha() for c in text)  
  capitalized_letters = sum(c.isupper() for c in text)  
  if total_letters == 0:
    return 0
  else:
    return (capitalized_letters / total_letters) * 100

if __name__ == "__main__":
    main()