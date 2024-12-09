import sys
import openai
import argparse

sys.path.insert(0, 'src')
from pipelines import run_baseline, run_multi_stage


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM Semantic Compression Script"
    )

    # must-have targets
    parser.add_argument(
        "-d", "--data", 
        type=str, help="dataset name",
        default='nyt'
    )
    parser.add_argument(
        "-m", "--model", type=str, choices=['baseline', 'multi-stage'],
        help="compression pipeline to run", default='multi-stage'
    )

    # data details
    parser.add_argument(
        "-s", "--samples", 
        type=int, 
        help="The number of documents to randomly sample from data. Reduces runtime.", 
        default=1000
    )
    parser.add_argument(
        "-r", "--random_seed", 
        type=int, 
        help="Random seed for reproducibility", 
        default=42
    )

    return parser.parse_args()


if __name__ == "__main__":
    # parse command-line arguments
    args = parse()

    # test data target
    assert args.data in ['nyt', 'yelp'], "Only support `nyt` and 'yelp` for now"

    # ask for openai API Key
    print("Please make sure you have a valid OpenAI API Key to run `gpt-4o-mini` inference")
    API_KEY = input("Enter OpenAI API Key: ")
    print()
    client = openai.OpenAI(api_key=API_KEY)

    # select pipeline to run
    if args.model == 'baseline':
        run_baseline(client, dataset_name=args.data, n=args.samples, seed=args.random_seed)
    elif args.model == 'multi-stage':
        # ask for text genre
        text_genre = input("Enter Text Genre: ")
        print()
        run_multi_stage(
            client, text_genre=text_genre, dataset_name=args.data,
            n=args.samples, seed=args.random_seed
        )
