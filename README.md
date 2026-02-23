# kg-gen: Knowledge Graph Generation from Any Text

Welcome! `kg-gen` helps you extract knowledge graphs from any plain text using AI. It can process both small and large text inputs, and it can also handle messages in a conversation format.

Why generate knowledge graphs? `kg-gen` is great if you want to:
- Create a graph to assist with RAG (Retrieval-Augmented Generation)
- Create graph synthetic data for model training and testing
- Structure any text into a graph 
- Analyze the relationships between concepts in your source text

We support API-based and local model providers via [LiteLLM](https://docs.litellm.ai/docs/providers), including OpenAI, Ollama, Anthropic, Gemini, Deepseek, and others. We also use [DSPy](https://dspy.ai/) for structured output generation.

## Powered by a model of your choice

Examples of models you can pass in:
- `ollama/llama3.3`
- `ollama/gpt-oss:120b`

Potentially, more models are available out of the box, however, for our MEMORISE use case we want to only use the models from our SDU infrastructure.

## Quick start

### How to install

Follow these steps to get `kg-gen` running on your local machine.

**1. Clone the Project**

First, clone the repository to your local machine and navigate into the directory:
```bash
git clone <your-repository-url>
cd kg-gen
```

**2. Configure the Environment with `uv`**

We recommend using `uv` for managing the Python environment and dependencies.

If you don't have `uv`, install it first:
```bash
pip install uv
```

Create and activate a virtual environment:
```bash
# Create a virtual environment in a .venv folder
uv venv

# Activate it (on macOS/Linux)
source .venv/bin/activate

# On Windows, use: .venv\Scripts\activate
```

Install the required packages (assuming a `requirements.txt` file exists):
```bash
uv pip install -r requirements.txt
```

**3. Set Environment Variables**

Create a `.env` file in the project's root directory by copying the file `.env.example`.

Add your keys to the `.env` file like this:
```.env
# Example .env file
USERNAME=my.user
PASSWORD=my.password
```

Load these variables into your current shell session by running:
```bash
source .env
```
This command "pushes" the variables from the `.env` file into your environment, making them available to the application. You'll need to do this for each new terminal session.


### How to run on MEMORISE data sample

Place your data to `./memo/data/sample_items.json`. Next, run `python3 ./memo/run_memo.py`. Look into the file if you want to feed another dataset or prepare the data before processing.


### How to use

Then import and use `kg-gen`. You can provide your text input in one of two formats:
1. A single string  
2. A list of Message objects (each with a role and content)

Below are some example snippets:
```python
from kg_gen import KGGen

# Initialize KGGen with optional configuration
kg = KGGen(
  model="openai/gpt-4o",  # Default model
  temperature=0.0,        # Default temperature
  api_key="YOUR_API_KEY"  # Optional if set in environment or using a local model
)

# EXAMPLE 1: Single string with context
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
graph_1 = kg.generate(
  input_data=text_input,
  context="Family relationships"
)
# Output: 
# entities={'Linda', 'Ben', 'Andrew', 'Josh'} 
# edges={'is brother of', 'is father of', 'is mother of'} 
# relations={('Ben', 'is brother of', 'Josh'), 
#           ('Andrew', 'is father of', 'Josh'), 
#           ('Linda', 'is mother of', 'Josh')}
```

### Visualizing KGs
```python
KGGen.visualize(graph, output_path, open_in_browser=True)
```

### More Examples - chunking, clustering, passing in a messages array 

```python
# EXAMPLE 2: Large text with chunking and clustering
with open('large_text.txt', 'r') as f:
  large_text = f.read()
  
# Example input text:
# """
# Neural networks are a type of machine learning model. Deep learning is a subset of machine learning
# that uses multiple layers of neural networks. Supervised learning requires training data to learn
# patterns. Machine learning is a type of AI technology that enables computers to learn from data.
# AI, also known as artificial intelligence, is related to the broader field of artificial intelligence.
# Neural nets (NN) are commonly used in ML applications. Machine learning (ML) has revolutionized
# many fields of study.
# ...
# """

graph_2 = kg.generate(
  input_data=large_text,
  chunk_size=5000,  # Process text in chunks of 5000 chars
  cluster=True      # Cluster similar entities and relations
)
# Output:
# entities={'neural networks', 'deep learning', 'machine learning', 'AI', 'artificial intelligence', 
#          'supervised learning', 'unsupervised learning', 'training data', ...} 
# edges={'is type of', 'requires', 'is subset of', 'uses', 'is related to', ...} 
# relations={('neural networks', 'is type of', 'machine learning'),
#           ('deep learning', 'is subset of', 'machine learning'),
#           ('supervised learning', 'requires', 'training data'),
#           ('machine learning', 'is type of', 'AI'),
#           ('AI', 'is related to', 'artificial intelligence'), ...}
# entity_clusters={
#   'artificial intelligence': {'AI', 'artificial intelligence'},
#   'machine learning': {'machine learning', 'ML'},
#   'neural networks': {'neural networks', 'neural nets', 'NN'}
#   ...
# }
# edge_clusters={
#   'is type of': {'is type of', 'is a type of', 'is a kind of'},
#   'is related to': {'is related to', 'is connected to', 'is associated with'
#  ...}
# }

# EXAMPLE 3: Messages array
messages = [
  {"role": "user", "content": "What is the capital of France?"}, 
  {"role": "assistant", "content": "The capital of France is Paris."}
]
graph_3 = kg.generate(input_data=messages)
# Output: 
# entities={'Paris', 'France'} 
# edges={'has capital'} 
# relations={('France', 'has capital', 'Paris')}

# EXAMPLE 4: Combining multiple graphs
text1 = "Linda is Joe's mother. Ben is Joe's brother."

# Input text 2: also goes by Joe."
text2 = "Andrew is Joseph's father. Judy is Andrew's sister. Joseph also goes by Joe."

graph4_a = kg.generate(input_data=text1)
graph4_b = kg.generate(input_data=text2)

# Combine the graphs
combined_graph = kg.aggregate([graph4_a, graph4_b])

# Optionally cluster the combined graph
clustered_graph = kg.cluster(
  combined_graph,
  context="Family relationships"
)
# Output:
# entities={'Linda', 'Ben', 'Andrew', 'Joe', 'Joseph', 'Judy'} 
# edges={'is mother of', 'is father of', 'is brother of', 'is sister of'} 
# relations={('Linda', 'is mother of', 'Joe'),
#           ('Ben', 'is brother of', 'Joe'),
#           ('Andrew', 'is father of', 'Joe'),
#           ('Judy', 'is sister of', 'Andrew')}
# entity_clusters={
#   'Joe': {'Joe', 'Joseph'},
#   ...
# }
# edge_clusters={ ... }
```

### Reference

This repo was forked from [kg-gen at Github](https://github.com/stair-lab/kg-gen). See the original repo for more instructions.
