import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import evaluate

# Load data
df = pd.read_csv('impression_300_llm.csv')

# Prepare data for model
df = df[['Report Name', 'History', 'Observation', 'Impression']]
df['input'] = df['Report Name'] + " " + df['History'] + " " + df['Observation']
df = df[['input', 'Impression']]

# Split dataset (train: 300, eval: 30)
train_data = df.iloc[:300]
eval_data = df.iloc[300:]

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)

# Load pre-trained model and tokenizer
model_name = "gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Fine-tune model with Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

"""Evaluation.py"""
# Load metrics
perplexity_metric = evaluate.load("perplexity")
rouge_metric = evaluate.load("rouge")

# Generate predictions on eval data
eval_preds = trainer.predict(eval_dataset)

# Calculate Perplexity
perplexity = perplexity_metric.compute(predictions=eval_preds.predictions)

# Calculate ROUGE Score
references = eval_data['Impression'].tolist()
predictions = tokenizer.batch_decode(eval_preds.predictions, skip_special_tokens=True)
rouge_score = rouge_metric.compute(predictions=predictions, references=references)

# Print Results
print(f"Perplexity: {perplexity['perplexity']}")
print(f"ROUGE Score: {rouge_score}")


"""Text_analysis.py"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stop words, stemmer, and lemmatizer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocess text
def preprocess_text(text):
    # Lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize and remove stop words
    tokens = [word for word in text.split() if word not in stop_words]
    
    # Stemming and Lemmatization
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing to dataset
df['processed_text'] = df['input'].apply(preprocess_text)


"""Convert Text into Embeddings and Find Top Word Pairs"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_text'])

# Calculate cosine similarity between word pairs
similarity_matrix = cosine_similarity(tfidf_matrix.T)
words = vectorizer.get_feature_names_out()

# Get top 100 word pairs based on similarity
similarities = []
for i in range(len(words)):
    for j in range(i+1, len(words)):
        similarities.append((words[i], words[j], similarity_matrix[i][j]))

# Sort word pairs by similarity
top_100_pairs = sorted(similarities, key=lambda x: x[2], reverse=True)[:100]


"""Visualization"""
import matplotlib.pyplot as plt
import networkx as nx

# Create a graph using the top 100 word pairs
G = nx.Graph()
for word1, word2, sim in top_100_pairs:
    G.add_edge(word1, word2, weight=sim)

# Plot the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True, node_size=50, font_size=10)
plt.show()


"""Interactive Visualization"""
import plotly.graph_objects as go
import networkx as nx

# Create NetworkX graph and set up Plotly visualization
G = nx.Graph()
for word1, word2, sim in top_100_pairs:
    G.add_edge(word1, word2, weight=sim)

pos = nx.spring_layout(G)

# Create edge traces for visualization
edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_trace['x'] += [x0, x1, None]
    edge_trace['y'] += [y0, y1, None]

# Create node traces
node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers+text',
    hoverinfo='text',
    marker=dict(
        showscale=True,
        colorscale='YlGnBu',
        size=10,
        color=[],
    ))

for node in G.nodes():
    x, y = pos[node]
    node_trace['x'] += [x]
    node_trace['y'] += [y]
    node_trace['text'] += [node]
    node_trace['marker']['color'] += [len(G[node])]

# Create Plotly figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False))
               )

fig.show()
