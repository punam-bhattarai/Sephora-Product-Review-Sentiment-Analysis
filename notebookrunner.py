import nbformat
from nbconvert import PythonExporter
import os

def load_notebook(path):
    # Load the notebook content
    with open(path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Convert the notebook to a Python script
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)
    
    # Create a dictionary to serve as the notebook's execution environment
    notebook_env = {}
    
    # Execute the notebook's code in the environment
    exec(source, notebook_env)
    
    return notebook_env

# Path to your notebook
notebook_path = os.path.join(os.path.dirname(__file__), 'sephora_new.ipynb')
notebook_env = load_notebook(notebook_path)

# Save necessary functions into variables
preprocess_text = notebook_env.get('preprocess_text')
get_sentiment_scores = notebook_env.get('get_sentiment_scores')
average_word_embedding = notebook_env.get('average_word_embedding')

print("preprocess_text:", preprocess_text)
print("get_sentiment_scores:", get_sentiment_scores)
print("average_word_embedding:", average_word_embedding)
