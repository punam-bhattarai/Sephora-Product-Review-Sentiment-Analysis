import re
import numpy as np
import pandas as pd
import ssl

import shap
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from textblob import TextBlob
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline, make_pipeline
from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings('ignore')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# Download necessary resources for NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

review_df_01 = pd.read_csv('/Users/punam/Desktop/Sephora/reviews_0-250.csv', index_col = 0, dtype={'author_id':'str'})
review_df_02 = pd.read_csv('/Users/punam/Desktop/Sephora//reviews_250-500.csv', index_col = 0, dtype={'author_id':'str'})
review_df_03 = pd.read_csv('/Users/punam/Desktop/Sephora/reviews_500-750.csv', index_col = 0, dtype={'author_id':'str'})
review_df_04 = pd.read_csv('/Users/punam/Desktop/Sephora//reviews_750-1250.csv', index_col = 0, dtype={'author_id':'str'})
review_df_05 = pd.read_csv('/Users/punam/Desktop/Sephora//reviews_1250-end.csv', index_col = 0, dtype={'author_id':'str'})
abbreviation_mapping_df = pd.read_csv('/Users/punam/Desktop/Sephora/slangs.csv')

# MERGIG ALL REVIEWS DATAFRAMES
review_df = pd.concat([review_df_01, review_df_02, review_df_03, review_df_04, review_df_05], axis=0)
review_df_subset =review_df.iloc[:10000]

# Create a dictionary from the DataFrame
abbreviation_mapping = dict(zip(abbreviation_mapping_df['Abbr'], abbreviation_mapping_df['Fullform']))

# Function to find abbreviations using regex
def find_abbreviations(s):
    return re.findall(r'\b[A-Z]{2,}\b', s)  # Ensure the regular expression pattern is balanced

review_df_subset['review_text'] = review_df_subset['review_text'].fillna('')

# Apply the function to extract abbreviations
review_df_subset['abbreviations'] = review_df_subset['review_text'].apply(find_abbreviations)
num_data_with_abbreviations = review_df_subset[review_df_subset['abbreviations'].apply(len) > 0].shape[0]

#Function to replace abbreviations with their full forms
# def replace_abbreviations(text):
#     if pd.isna(text):  # Check if the text is NaN
#         return text  # Return NaN if it is NaN
#     for abbr, full_form in abbreviation_mapping.items():
#         if isinstance(abbr, str) and isinstance(full_form, str):  # Check if abbr and full_form are strings
#             text = text.replace(abbr, full_form)
#    return text

# Apply the function to the 'text' column of the DataFrame
#review_df_subset['text'] = review_df_subset['text'].apply(replace_abbreviations)

# Custom function to handle contractions
contraction_map = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mightn't": "might not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}
def expand_contractions(text, contraction_mapping=contraction_map):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        if expanded_contraction:
            expanded_contraction = first_char + expanded_contraction[1:]
        else:
            expanded_contraction = match  # Return original match if expansion not found
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# Custom function to handle negations (e.g., not happy -> not_happy)
def handle_negations(text):
    negations = ['no', 'not', 'none', 'neither', 'never', 'nobody', 'nothing']
    words = text.split()
    for i in range(len(words)):
        if words[i] in negations:
            if i < len(words) - 1:
                words[i + 1] = 'not_' + words[i + 1]
    return ' '.join(words)

# Custom function to preprocess text
def preprocess_text(text):
    # Handle contractions
    text = expand_contractions(text)

    # Handle negations
    text = handle_negations(text)

    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the words to their base form
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


# Apply preprocessing to the text column in your DataFrame
review_df_subset['preprocessed_text'] = review_df_subset['review_text'].apply(preprocess_text)

# Initialize the sentiment analyzer
#sid = SentimentIntensityAnalyzer()

# Function to assign sentiment scores to each review
def calculate_sentiment_score(text):
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get the sentiment polarity
    sentiment_score = blob.sentiment.polarity

    # Assign sentiment label based on polarity
    if sentiment_score > 0:
        return 'positive'
    elif sentiment_score < 0:
        return 'negative'
    else:
        return 'neutral'


# Apply sentiment analysis to each review in the DataFrame
review_df_subset['sentiment'] = review_df_subset['preprocessed_text'].apply(calculate_sentiment_score)

tokenized_sentences = [review_text.split() for review_text in review_df_subset['preprocessed_text']]

# Train Word2Vec model
word2vec_model = Word2Vec(tokenized_sentences, min_count=1)

def count_words(text):
    return len(text.split())

# Function to calculate the number of sentences in each review
def count_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)

# Function to perform sentiment analysis and get sentiment scores
def get_sentiment_scores(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity
def average_word_embedding(text):
    tokens = text.split()
    embeddings = []
    for token in tokens:
        if token in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[token])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)  # Return zero vector if no embeddings found

# Apply the function to each review
review_df_subset['review_length'] = review_df_subset['preprocessed_text'].apply(count_words)
review_df_subset['num_sentences'] = review_df_subset['preprocessed_text'].apply(count_sentences)
review_df_subset['polarity'], review_df_subset['subjectivity'] = zip(*review_df_subset['preprocessed_text'].apply(get_sentiment_scores))
review_df_subset['word_embeddings'] = review_df_subset['preprocessed_text'].apply(average_word_embedding)

#review_df_subset.to_csv('/Users/shweta/Downloads/review_df_subset_embedding.csv', index=False)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(review_df_subset['preprocessed_text'])


# Step 1: Split the dataset into training and testing sets
X = review_df_subset['preprocessed_text']  # Text data
y = review_df_subset['sentiment']  # Sentiment labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Feature Extraction using TF-IDF
#tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 3: Choose appropriate algorithm (Naive Bayes)
model = MultinomialNB()

# Step 4: Train the model
model.fit(X_train_tfidf, y_train)

# Step 5: Evaluate the performance of the model
y_pred = model.predict(X_test_tfidf)

# negative_indices = [i for i, sentiment in enumerate(y_pred) if sentiment == 'positive']
# negative_data = X_test.iloc[negative_indices]
#
# print("negative_indices",negative_indices)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted',zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Naive Bayes Accuracy:", accuracy)
print("Naive Bayes Precision:", precision)
print("Naive Bayes Recall:", recall)
print("Naive Bayes F1-score:", f1)

# Step 3: Choose Support Vector Machines (SVM) algorithm
svm_model = SVC(kernel='linear')

# Step 4: Train the SVM model
svm_model.fit(X_train_tfidf, y_train)

# Step 5: Evaluate the performance of the model
y_pred_svm = svm_model.predict(X_test_tfidf)

# Calculate evaluation metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='weighted',zero_division=1)
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# Print the evaluation metrics for SVM
print("SVM Accuracy:", accuracy_svm)
print("SVM Precision:", precision_svm)
print("SVM Recall:", recall_svm)
print("SVM F1-score:", f1_svm)

# Topic modeling with LDA
lda_model = LatentDirichletAllocation(n_components=6, random_state=42)
lda_topics = lda_model.fit_transform(tfidf_matrix)

# Visualize topics with word clouds
def visualize_topics(lda_model, feature_names, n_words=20):
    num_topics = len(lda_model.components_)
    n_cols = min(2, num_topics)  # Maximum of 2 columns
    n_rows = -(-num_topics // n_cols)  # Ceiling division to determine rows
    for idx, topic in enumerate(lda_model.components_):
        # Get top words for each topic
        top_words_idx = topic.argsort()[:-n_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]

        # Create word cloud for each topic
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_words))

        # Plot word cloud
        #plt.figure(figsize=(10, 5))
        plt.subplot(n_rows, n_cols, idx + 1)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Topic {idx + 1}')
        plt.axis('off')

    # Show all word clouds
    plt.tight_layout()
    plt.show()
# Visualize topics using word clouds
visualize_topics(lda_model, tfidf_vectorizer.get_feature_names_out())


# # Calculate average sentiment score for each topic
topic_sentiment = []
for topic_idx, topic in enumerate(lda_topics[:10]):
    top_reviews_idx = topic.argsort()[-10:]  # Example: Top 10 reviews for each topic
    topic_reviews = review_df_subset.iloc[top_reviews_idx]
    avg_sentiment = topic_reviews['polarity'].mean()
    topic_sentiment.append(avg_sentiment)

# Visualize sentiment scores for topics
plt.figure(figsize=(8, 5))
plt.bar(range(len(topic_sentiment)), topic_sentiment, color='skyblue')
plt.xlabel('Topic')
plt.ylabel('Average Sentiment Score')
plt.title('Sentiment Scores for Topics')
plt.xticks(range(len(topic_sentiment)), [f'Topic {i+1}' for i in range(len(topic_sentiment))],rotation=45)
plt.show()

def calculate_similarity(lda_model, feature_names, words_or_phrases):
    similarities = []
    # Iterate over each word or phrase
    for word_or_phrase in words_or_phrases:
        # Calculate cosine similarity with each topic
        similarity_with_topics = []
        for topic in lda_model.components_:
            # Get top words for the topic
            top_words_idx = topic.argsort()[:-len(topic) - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            # Calculate cosine similarity between word_or_phrase and top words of the topic
            similarity = cosine_similarity([word2vec_model.wv[word_or_phrase]], [word2vec_model.wv[word] for word in top_words])
            similarity_with_topics.append(similarity[0][0])  # Extracting the scalar value from the array
        similarities.append(similarity_with_topics)
    return np.array(similarities)

# Example words or phrases for semantic relationship exploration
words_or_phrases = [
    "moisturizer", "foundation", "lipstick", "fragrance", "serum", "hydrating", "matte", "primer","concealer", "mascara", "sunscreen","cream",
    "exfoliating", "glow", "texture", "vegan", "organic", "natural", "blemish", "radiant", "tone", "shade", "coverage", "pigmented",
    "lightweight", "smooth"
]

# Calculate similarities
similarities = calculate_similarity(lda_model, tfidf_vectorizer.get_feature_names_out(), words_or_phrases)
n_topics = 6
# Visualize the semantic relationship using a heatmap

plt.figure(figsize=(10, 6))
sns.heatmap(similarities, annot=True, xticklabels=[f"Topic {i+1}" for i in range(n_topics)], yticklabels=words_or_phrases, cmap="YlGnBu")
plt.title("Semantic Relationship between Words/Phrases and LDA Topics")
plt.xlabel("LDA Topics")
plt.ylabel("Words/Phrases")
plt.show()

# #  Model Selection and Hyperparameter Tuning (SVM)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)
best_params = grid_search.best_params_

# Train the model with best hyperparameters
svm_model = SVC(**best_params)
svm_model.fit(X_train_tfidf, y_train)
print("Best parameters SVM:", grid_search.best_params_)
# Evaluate the performance of the model
y_pred = svm_model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy SVM:", accuracy)

# Define the pipeline with TF-IDF vectorizer and Multinomial Naive Bayes classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Define parameter grid for grid search
param_grid = {
    'tfidf__max_features': [1000, 5000, 10000],  # Maximum number of features for TF-IDF vectorizer
    'tfidf__ngram_range': [(1, 1), (1, 2)],       # N-gram range for TF-IDF vectorizer
    'clf__alpha': [0.1, 0.5, 1.0],                # Alpha parameter for Multinomial Naive Bayes
}

# Initialize GridSearchCV with the pipeline, parameter grid, and cross-validation strategy
grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True), scoring='accuracy', verbose=1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found by the grid search
print("Best parameters Naive:", grid_search.best_params_)

# Evaluate the best model on the test data
best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print("Accuracy on test set for Naive bayes:", accuracy)

# # LIME Explanation

pipeline = make_pipeline(tfidf_vectorizer, model)
class_names = ['negative', 'positive','neutral']
explainer = LimeTextExplainer(class_names=class_names)
#print(X_test.index)
#index = [1501, 2586, 2653, 1055,  705,  106,  589, 2468, 2413, 1600]
#n_in = 1221, 1262, 1562, 1777, 2030, 2668, 2758, 2847, 3127
index=30
text = X_test.iloc[index]
exp = explainer.explain_instance(text, pipeline.predict_proba, num_features=6)
with open(f"data_{index}.html", "w") as file:
    file.write(exp.as_html())

review_df_subset['review_text'] = review_df_subset['review_text'].astype(str)
#
# # Load pre-trained SentenceTransformer model
shap_model = SentenceTransformer('bert-base-nli-mean-tokens')
#
# # Generate embeddings for the reviews
review_df_subset['embedding'] = review_df_subset['review_text'].apply(lambda x: shap_model.encode(x))

# Create a new DataFrame with embeddings
embeddings = pd.DataFrame(review_df_subset['embedding'].tolist())

X_train, X_test, y_train, y_test = train_test_split(embeddings, review_df_subset['sentiment'], test_size=0.2, random_state=42)
#
# # Train a classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

X_sub = shap.sample(X_train, 100)
explainer = shap.Explainer(classifier.predict_proba, X_sub)
max_evals = 2 * X_train.shape[1] + 1
shap_values = explainer(X_test.iloc[0:100], max_evals=max_evals)
shap_values.feature_names = [str(name) for name in shap_values.feature_names]
class_index = 1
data_index = 1
shap.plots.waterfall(shap_values[data_index][:, class_index])
#
#Perform PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(tfidf_matrix.toarray())

# Perform t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(tfidf_matrix.toarray())

# Define a color map for sentiments
color_map = {'positive': 'Green', 'negative': 'red', 'neutral': 'Yellow'}

# Plot PCA result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=review_df_subset['sentiment'].map(color_map))
plt.title('PCA')
#plt.savefig('/Users/shweta/Downloads/Anime_3014/PCA.png')

# Plot t-SNE result
plt.subplot(1, 2, 2)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=review_df_subset['sentiment'].map(color_map))
plt.title('t-SNE')
#plt.savefig('/Users/shweta/Downloads/Anime_3014/tSNE.png')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(review_df_subset['sentiment'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

sns.set_style("whitegrid")

# Plot histogram of sentiment scores
plt.figure(figsize=(10, 6))
sns.histplot(review_df_subset['rating'], bins=20, color='skyblue', kde=True)
plt.title('Distribution of Sentiment Ratings')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Plot box plot of sentiment scores
plt.figure(figsize=(8, 6))
sns.boxplot(y=review_df_subset['rating'], color='lightblue')
plt.title('Box Plot of Sentiment rating')
plt.ylabel('Sentiment rating')
plt.show()


review_df_subset.to_csv('/Users/shweta/Downloads/review_df_final1.csv', index=False)

