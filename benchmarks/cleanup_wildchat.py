from transformers import AutoTokenizer
import click
import os
from tqdm.auto import tqdm
import requests
import pandas as pd

BASE_URL = "https://huggingface.co/datasets/allenai/WildChat-1M/resolve/main/data/train-{i:05d}-of-00014.parquet"
NUM_FILES = 14
DATA_DIR = "./wildchat_data"
COMBINED_JSON_PATH = "./ShareGPT.json"

@click.command()
@click.option("--model", required=True, type=str)
@click.option("--wild_chat_path",
              default=COMBINED_JSON_PATH,
              type=click.Path(dir_okay=False, file_okay=True), )
def main(model: str, wild_chat_path: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    all_data = []

    for i in range(NUM_FILES):
        file_url = BASE_URL.format(i=i)
        file_path = file_url.split("/")[-1]

        if not os.path.exists(file_path):
            print(f"{file_path} not found. Downloading from Hugging Face...")
            response = requests.get(file_url)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {file_path}.")

        df = pd.read_parquet(file_path)
        all_data.append(df)
    df = pd.concat( all_data )

    print(f"Loaded {len(all_data)} conversations.")
    tokenizer = AutoTokenizer.from_pretrained(model, token=os.getenv("HFAPI_TOKEN"))

    df[ 'num_round' ] = df[ 'turn' ] * 2
    for chat in tqdm(df):
        for message in chat['conversations']:
            message['num_tokens'] = max(1, len(tokenizer.tokenize(message['value'])))

    df.to_parquet( wild_chat_path )
    print(f"Saved combined parquet to {wild_chat_path}")


if __name__ == "__main__":
    main()
