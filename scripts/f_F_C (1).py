# Import required libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Load the feedback data (replace with your file path)
fact_feedback = pd.read_csv("C:/procsv/pp/ieact_[feedback.csv")

# Display the first few rows
print(fact_feedback.head())


# Preprocessing function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back to string
    return ' '.join(tokens)


# Apply preprocessing to the 'Comment' column (assuming it's in your dataset)
fact_feedback['Cleaned_Comment'] = fact_feedback['Comment'].apply(preprocess_text)

# Sentiment analysis using VADER
sia = SentimentIntensityAnalyzer()


# Function to get sentiment scores
def get_sentiment(text):
    return sia.polarity_scores(text)


# Apply sentiment analysis to the cleaned comments
fact_feedback['Sentiment'] = fact_feedback['Cleaned_Comment'].apply(get_sentiment)

# Separate sentiment into individual columns (positive, negative, neutral, compound)
fact_feedback = pd.concat([fact_feedback.drop(['Sentiment'], axis=1), fact_feedback['Sentiment'].apply(pd.Series)],
                          axis=1)

# Display the first few rows after processing
print(fact_feedback.head())

# Save the processed data to a new CSV
fact_feedback.to_csv("C:/procsv/pp/ieact_[feedback.csv", index=False)

