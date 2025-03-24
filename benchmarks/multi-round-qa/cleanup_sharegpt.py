import json
from transformers import AutoTokenizer
import click
import os
from tqdm.auto import tqdm


@click.command( )
@click.option( "--model", required = True, type = str )
@click.option( "--share_gpt_path",
               default = "./ShareGPT.json",
               type = click.Path( dir_okay = False, file_okay = True ), )
def main( model: str, share_gpt_path: str ):
    with open( share_gpt_path, "r", encoding = "utf-8" ) as file:
        sharegpt_data = json.load( file )
    tokenizer = AutoTokenizer.from_pretrained( model, token = os.getenv( "HFAPI_TOKEN" ) )
    for chat in tqdm( sharegpt_data ):
        chat[ 'num_round' ] = len( chat[ 'conversations' ] )
        for message in chat[ 'conversations' ]:
            message[ 'num_tokens' ] = min( 1, len( tokenizer.tokenize( message[ 'value' ] ) ) )
    with open( share_gpt_path, "w", encoding = "utf-8" ) as file:
        json.dump( sharegpt_data, file, indent = 2 )


if __name__ == "__main__":
    main( )
