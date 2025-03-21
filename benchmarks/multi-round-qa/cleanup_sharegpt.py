import json
import tokenizers
import click


@click.command( )
@click.option( "--model", required = True, type = str )
@click.option( "--share_gpt_path",
               default = "./ShareGPT.json",
               type = click.Path( dir_okay = False, file_okay = True ), )
def main( model: str, share_gpt_path: str ):
    with open( share_gpt_path, "r", encoding = "utf-8" ) as file:
        sharegpt_data = json.load( file )
    for chat in sharegpt_data:
        chat[ 'num_round' ] = len( chat[ 'conversations' ] )
        for message in chat[ 'conversations' ]:
            # TODO: use tokenizer to compute number of tokens for that model
            message[ 'num_tokens' ] = 1
    with open( share_gpt_path, "w", encoding = "utf-8" ) as file:
        json.dump( sharegpt_data, file, indent = 2 )


if __name__ == "__main__":
    main( )
