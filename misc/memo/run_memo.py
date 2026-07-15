import json
from pathlib import Path

from kg_gen import KGGen
from memo.auth_handler import get_keycloak_token


def read_sample_items(
    input_path: str | Path = Path(__file__).parent / "data" / "sample_items.json",
):
    with open(input_path, "r") as f:
        data = json.load(f)["result"]
    output_text = "\n\n".join(
        f"**{point['title']}**\n{point['creator']}\n{point['text']}"
        for key, point in data.items()
    )
    # print(output_text)
    return output_text


if __name__ == "__main__":
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
    #
    sample_data = read_sample_items()
    graph_1 = kg.generate(
        input_data=sample_data,
        context="Heritage of Nazi Persecution",
    )
    print(graph_1)
