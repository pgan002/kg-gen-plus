import json
from pathlib import Path

from kg_gen import KGGen
import kg_gen
from memo.auth_handler import get_keycloak_token

def read_data(input_path: str | Path = Path(__file__).parent / "data" / "detections.json"):
    out_fields = {"image_file", "detections", "ImageURL", "URL"}
    with open(input_path, "r") as f:
        data = json.load(f)
    filter_data = [{k: v for k, v in data_point.items()
                    if k not in out_fields}
                   for data_point in data]
    # print(output_text)
    return filter_data

if __name__ == "__main__":
    # Initialize KGGen with the token
    kg = KGGen(
        # Use 'openai/' prefix to force standard HTTP client with Bearer token auth
        model="openai/gpt-oss:120b",
        temperature=0.3,
        # Append /v1 to access Ollama's OpenAI-compatible API
        api_base="https://ollama.dev.memorise.sdu.dk/v1",
        # Pass the token directly to KGGen
        api_key=get_keycloak_token()
    )
    #
    sample_data = read_data()
    # print(sample_data[0]["Caption"])
    # captions = [d.get("Caption", "") for d in sample_data]
    # print(f"{len([d for d in sample_data if 'Caption' in d]) = }")
    data_as_str = "\n\n".join([str(d) for d in sample_data[:25]])
    # print(f"{data_as_str[:200] = }")
    graph_1 = kg.generate(
        input_data=data_as_str,
        context="Heritage of Nazi Persecution",
        api_key=get_keycloak_token(),
        # deduplication_method=None
    )
    print(graph_1)
