from kg_gen import KGGen
from memo.auth_handler import get_keycloak_token

# Fetch the token first
keycloak_token = get_keycloak_token()

# Initialize KGGen with the token
kg = KGGen(
    # Use 'openai/' prefix to force standard HTTP client with Bearer token auth
    model="openai/gpt-oss:120b",
    temperature=0.3,
    # Append /v1 to access Ollama's OpenAI-compatible API
    api_base="https://ollama.dev.memorise.sdu.dk/v1",
    # Pass the token directly to KGGen
    api_key=keycloak_token,
)

# EXAMPLE 1: Single string with context
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
graph_1 = kg.generate(
    input_data=text_input,
    context="Family relationships",
)
print(graph_1)
# Output:
# entities={'Linda', 'Ben', 'Andrew', 'Josh'}
# edges={'is brother of', 'is father of', 'is mother of'}
# relations={('Ben', 'is brother of', 'Josh'),
#           ('Andrew', 'is father of', 'Josh'),
#           ('Linda', 'is mother of', 'Josh')}
