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
- `openai/gpt-5.4-mini`
- `ollama/gpt-oss:120b`

## Quick start

This application is containerized and can be run using Docker Compose.

**1. Configure the Environment**

Create a `.env` file in the project's root directory by copying the file `.env.example`.

Add your keys to the `.env` file like this:
```.env
# Example .env file
OPENAI_API_KEY="your_openai_api_key_here"
LLM_MODEL=openai/gpt-5.4-mini
LLM_TEMPERATURE=1.0
RETRIEVAL_MODEL=all-MiniLM-L6-v2
```

**2. Run the application**

To start the application, run the following command in the root of the project:

```bash
docker-compose up --build
```

This will build the Docker image and start the web server. The server provides:
- **Web APIs at `/api`**: for generating graphs from text.
- **Graphical User UI at `/ui`**: for visually investigating the graphs.

You can access the application at `http://localhost:5000`.

## How to use `kg-gen` in your code

You can also use `kg-gen` as a library in your own Python code.

First, install it:
```bash
pip install .
```

Then import and use `kg-gen`. You can provide your text input as a string.

Below is an example snippet:
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

# EXAMPLE: Single string with context
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
graph, stats = kg.generate(
  input_data=InputData(text=text_input, id="text_1"),
  entity_context="Family relationships"
)
print(graph)
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

The library also supports processing large texts by deduplicating entities and relations from multiple graphs.

### Reference

This repo was forked from [kg-gen at Github](https://github.com/stair-lab/kg-gen). See the original repo for more instructions.
