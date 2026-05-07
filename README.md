# SynthIE experiments

The target folder is [benchmarks](./benchmarks). You will find a script `run_synthie.py`. The KGGen instance is created inside the script.

You need to create your own `.env` file by copying [`.env.example`](./.env.example) and load it to your environment. Adapt the KGGen instance creation to indicate the correct LLM for your settings.

The SynthIE data should be placed inside `./benchmarks/data/synthie`. The data can be downloaded from [here](https://huggingface.co/datasets/martinjosifoski/SynthIE/tree/main/processed/sdg_code_davinci_002).

# kg-gen: Knowledge Graph Generation from Any Text

Welcome! `kg-gen` helps you extract knowledge graphs from any plain text using AI. It can process both small and large text inputs.

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

Where do I get the values? You need a [Keycloak](https://keycloak.dev.memorise.sdu.dk/) user credentials and client secret in the realm `oauth2-proxy`.

Load these variables into your current shell session by running:
```bash
source .env
```
This command "pushes" the variables from the `.env` file into your environment, making them available to the application. You'll need to do this for each new terminal session.


### How to run on MEMORISE data sample

Place your data to `./memo/data/sample_items.json`. Next, run `python3 ./memo/run_memo.py`. Look into the file if you want to feed another dataset or prepare the data before processing.


### How to use

Then import and use `kg-gen`. You can provide your text input as a string.

Below are some example snippets:
```python
from kg_gen import KGGen
from kg_gen.models import InputData
from kg_gen.steps import DeduplicateMethod

# Initialize KGGen with optional configuration
kg = KGGen(
  model="openai/gpt-4o",  # Default model
  temperature=0.0,        # Default temperature
  api_key="YOUR_API_KEY"  # Optional if set in environment or using a local model
)

# EXAMPLE 1: Single string with context
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
graph_1, stats_1 = kg.generate(
  input_data=InputData(text=text_input, id="text_1"),
  entity_context="Family relationships"
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

### More Examples - chunking, deduplication

```python
# EXAMPLE 2: Large text with chunking and deduplication
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

graph_2, stats_2 = kg.generate(
  input_data=InputData(text=large_text, id="large_text_1"),
  chunk_size=5000,  # Process text in chunks of 5000 chars
  deduplication_method=DeduplicateMethod.FULL # Deduplicate similar entities and relations
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

# EXAMPLE 3: Combining multiple graphs
text1 = "Linda is Joe's mother. Ben is Joe's brother."

# Input text 2: also goes by Joe."
text2 = "Andrew is Joseph's father. Judy is Andrew's sister. Joseph also goes by Joe."

graph3_a, _ = kg.generate(input_data=InputData(text=text1, id="text_a"))
graph3_b, _ = kg.generate(input_data=InputData(text=text2, id="text_b"))

# Combine the graphs
combined_graph = kg.aggregate([graph3_a, graph3_b])

# Optionally deduplicate the combined graph
deduplicated_graph, dedup_stats = kg.deduplicate(
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
