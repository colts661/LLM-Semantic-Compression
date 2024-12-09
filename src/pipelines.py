from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from data import load_downsample_data, save_data
from traditional import lossless_baselines, lossless_postprocess
from util import plot_text_length_distributions
from openai_generation import *

def run_baseline(client, dataset_name, n, seed):
    """
    Running the baseline method and evaluate on failure rate, CR, similarity
    """
    print(f"#### Running Baseline Method ####")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {MODEL}")
    print(f"Number of Samples: {n}")
    print(f"Random Seed: {seed}\n")

    # Load data
    df = load_downsample_data(dataset_name=dataset_name, n=n, random_seed=seed)

    # Run lossless baselines
    lossless_baselines(df)

    # Load prompts
    COMPRESS_BASELINE = load_prompt("compress-baseline")
    DECOMPRESS_BASELINE = load_prompt("decompress-baseline")
    
    print("#### Compress and Decompress ####")
    # Compress
    tqdm.pandas(desc=f"Baseline Compressor")
    text_compressed_baseline = df['sentence'].progress_apply(generate_text, args=(client, COMPRESS_BASELINE,))

    # Decompress
    tqdm.pandas(desc=f"Baseline Decompressor")
    text_decompressed_baseline = text_compressed_baseline.progress_apply(generate_text, args=(client, DECOMPRESS_BASELINE,))

    # save file
    baseline_processed = df.copy()[['sentence', 'label']].assign(
        compressed_baseline=text_compressed_baseline,
        decompressed_baseline=text_decompressed_baseline
    )
    save_data(baseline_processed, saved_name=f"{dataset_name}_baseline_processed")

    # baseline: run failure rate
    failure_rate = text_decompressed_baseline.str.startswith("I'm sorry").mean()
    print(f"Failure Rate: {failure_rate * 100:.1f}%")
    baseline_success = baseline_processed[~text_decompressed_baseline.str.startswith("I'm sorry")]
    
    # run compression ratio
    baseline_cr = (1 - baseline_success['compressed_baseline'].apply(len) / baseline_success['sentence'].apply(len)).mean()
    print(f"Compression Ratio: {baseline_cr * 100:.1f}%\n")

    # run sbert similarity
    print("#### Evaluating on Similarity ####")
    print("Computing SBERT Embeddings")
    st_model = SentenceTransformer("all-mpnet-base-v2")
    original_sentence_embeddings = st_model.encode(baseline_success['sentence'].tolist(), show_progress_bar=True)
    baseline_decompressed_sentence_embeddings = st_model.encode(baseline_success['decompressed_baseline'].tolist(), show_progress_bar=True)
    baseline_sim = st_model.similarity_pairwise(original_sentence_embeddings, baseline_decompressed_sentence_embeddings).mean()
    print(f"Decompressed SBERT Similarity: {round(baseline_sim.item(), 3)}")



def run_multi_stage(client, text_genre, dataset_name, n, seed):
    """
    Running the multi-stage pipeline and evaluate on CR, similarity, lossless postprocessing, and classification
    accuracy
    """
    print(f"#### Running Multi-Stage Compression Pipeline ####")
    print(f"Dataset: {dataset_name}")
    print(f"Text Genre: {text_genre}")
    print(f"Model: {MODEL}")
    print(f"Number of Samples: {n}")
    print(f"Random Seed: {seed}\n")

    # Load data
    df = load_downsample_data(dataset_name=dataset_name, n=n, random_seed=seed)

    # Run lossless baselines
    lossless_baselines(df)

    # Load Prompts
    COMPRESS_TO_FORMATTED = load_prompt("compress-to-formatted", text_genre)
    COMPRESS_TO_SHORT = load_prompt("compress-to-short", text_genre)
    DECOMPRESS_TO_FORMATTED = load_prompt("decompress-to-formatted", text_genre)
    DECOMPRESS_TO_TEXT = load_prompt("decompress-to-text", text_genre)

    # Compress
    print("#### Compress and Decompress ####")
    tqdm.pandas(desc=f"Compressor 1")
    text_compressed_stage_1 = df['sentence'].progress_apply(generate_text, args=(client, COMPRESS_TO_FORMATTED,))

    tqdm.pandas(desc=f"Compressor 2")
    text_compressed_stage_2 = text_compressed_stage_1.progress_apply(generate_text, args=(client, COMPRESS_TO_SHORT,))

    # Decompress
    tqdm.pandas(desc=f"Decompressor 2")
    text_decompressed_stage_2 = text_compressed_stage_2.progress_apply(generate_text, args=(client, DECOMPRESS_TO_FORMATTED,))

    tqdm.pandas(desc=f"Decompressor 1")
    text_decompressed_stage_1 = text_decompressed_stage_2.progress_apply(generate_text, args=(client, DECOMPRESS_TO_TEXT,))

    # save file
    if dataset_name == 'nyt':
        cols = ['sentence', 'label', 'label_fine']
    else:
        cols = ['sentence', 'label']
    
    multi_stage_processed = df.copy()[cols].assign(
        compressed_to_formatted=text_compressed_stage_1,
        compressed_to_short=text_compressed_stage_2,
        decompressed_to_formatted=text_decompressed_stage_2,
        decompressed_to_text=text_decompressed_stage_1
    )
    save_data(multi_stage_processed, saved_name=f"{dataset_name}_multi_stage_processed")

    # run compression ratio
    after_1_cr = (1 - multi_stage_processed['compressed_to_formatted'].apply(len) / multi_stage_processed['sentence'].apply(len)).mean()
    after_2_cr = (1 - multi_stage_processed['compressed_to_short'].apply(len) / multi_stage_processed['sentence'].apply(len)).mean()
    print("Compression Ratio:")
    print(f"After Stage 1: {after_1_cr * 100:.1f}%")
    print(f"After Stage 2: {after_2_cr * 100:.1f}%")

    # show text length distributions
    dist_fig = plot_text_length_distributions(dataset_name=dataset_name, df=multi_stage_processed)
    dist_fig.savefig(f'data/{dataset_name}_compressed_length_dist.png', dpi=300)
    print("Text Length Distribution Figure Saved\n")

    # run sbert similarity
    print("#### Evaluating on Similarity ####")
    print("Computing SBERT Embeddings")
    st_model = SentenceTransformer("all-mpnet-base-v2")
    original_sentence_embeddings = st_model.encode(multi_stage_processed['sentence'].tolist(), show_progress_bar=True)
    decompressed_sentence_embeddings = st_model.encode(multi_stage_processed['decompressed_to_text'].tolist(), show_progress_bar=True)
    mid_decompressed_sentence_embeddings = st_model.encode(multi_stage_processed['decompressed_to_formatted'].tolist(), show_progress_bar=True)

    after_1_sim = st_model.similarity_pairwise(original_sentence_embeddings, mid_decompressed_sentence_embeddings).mean()
    after_2_sim = st_model.similarity_pairwise(original_sentence_embeddings, decompressed_sentence_embeddings).mean()

    print("SBERT Similarity:")
    print(f"After Stage 1: {round(after_1_sim.item(), 3)}%")
    print(f"After Stage 2: {round(after_2_sim.item(), 3)}%\n")

    # run lossless post-processing
    lossless_postprocess(multi_stage_processed)

    # evaluate on downstream classification
    print("#### Evaluating on Classification ####")
    if dataset_name == 'yelp':
        CLASSIFIER = load_classification_prompt(text_genre, task="classify-cuisine")
    else:
        CLASSIFIER = load_classification_prompt(text_genre)  # default to topic
    
    run_classification(
        client=client,
        classify_prompt=CLASSIFIER,
        df=multi_stage_processed,
        label_col='label'
    )

    # special: NYT fine-grained labels
    if dataset_name == 'nyt':
        run_classification(
            client=client,
            classify_prompt=CLASSIFIER,
            df=multi_stage_processed,
            label_col='label_fine'
        )
