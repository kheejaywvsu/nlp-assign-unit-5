import wikipedia
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet', 'stopwords'])

def fetch_wikipedia_docs(topics):

    documents, titles = [], []
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    for topic in topics:
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            text = re.sub(r'[^a-zA-Z\s]', '', page.summary).lower()
            tokens = nltk.word_tokenize(text)
            processed = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
            documents.append(" ".join(processed))
            titles.append(page.title)
        except:
            documents.append("")
            titles.append(f"{topic} (Error)")
    return documents, titles
