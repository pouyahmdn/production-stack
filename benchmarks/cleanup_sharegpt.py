import json
from transformers import AutoTokenizer
import click
import os
from tqdm.auto import tqdm
import requests

DEFAULT_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

SYSTEM_PROMPT = ("You are a knowledgeable, efficient, and direct AI assistant. "
                             "Provide concise answers, focusing on the key information needed. "
                             "Offer suggestions tactfully when appropriate to improve outcomes. "
                             "Engage in productive collaboration with the user.\n\n")
@click.command( )
@click.option( "--model", required = True, type = str )
@click.option( "--share_gpt_path",
               default = "./ShareGPT.json",
               type = click.Path( dir_okay = False, file_okay = True ), )
def main( model: str, share_gpt_path: str ):
    if not os.path.exists( share_gpt_path + ".raw" ):
        print( f"{share_gpt_path}.raw not found. Downloading from Hugging Face..." )
        response = requests.get( DEFAULT_URL )
        response.raise_for_status( )
        with open( share_gpt_path + ".raw", "w", encoding = "utf-8" ) as file:
            file.write( response.text )
        print( "Download complete." )

    if not os.path.exists( share_gpt_path ):
        with open( share_gpt_path + ".raw", "r", encoding = "utf-8" ) as file:
            sharegpt_data = json.load( file )
        tokenizer = AutoTokenizer.from_pretrained( model, token = os.getenv( "HFAPI_TOKEN" ) )
        assert len(tokenizer.tokenize(SYSTEM_PROMPT)) == 42, 'Arash: You need to update the system prompt length here and also in multi-round-qa.py'
        roles = [ 'human', 'gpt' ]
        to_keep = set( )
        for i, chat in tqdm( enumerate( sharegpt_data ), total = len( sharegpt_data ) ):
            to_keep.add( i )
            chat[ 'num_round' ] = len( chat[ 'conversations' ] )
            for j, message in enumerate( chat[ 'conversations' ] ):
                message[ 'num_tokens' ] = max( 1, len( tokenizer.tokenize( message[ 'value' ] ) ) )
                if message[ 'from' ] != roles[ j % 2 ]:
                    to_keep.discard( i )
        print(f'Only keeping {len(to_keep)}/{len(sharegpt_data)} conversations due to irregular conversation roles...')
        sharegpt_data = [sharegpt_data[ i ] for i in to_keep]
        with open( share_gpt_path, "w", encoding = "utf-8" ) as file:
            json.dump( sharegpt_data, file, indent = 2 )


if __name__ == "__main__":
    main( )
