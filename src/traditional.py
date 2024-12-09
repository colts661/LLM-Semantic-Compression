import zlib
import bz2
from tqdm import tqdm

def compress_gzip(input_str):
    """
    Compress a single document using gzip. Lossless Compression
    """
    input_bytes = input_str.encode('utf-8')
    return zlib.compress(input_bytes)

def compress_bz2(input_str):
    """
    Compress a single document using bz2. Lossless Compression
    """
    input_bytes = input_str.encode('utf-8')
    return bz2.compress(input_bytes)


def lossless_baselines(df):
    """
    Perform both lossless algorithms as baseline
    """
    print("#### Running Lossless Baselines ####")
    df = df.copy()

    # apply gzip
    tqdm.pandas(desc="Applying gzip lossless baseline")
    df['gzip'] = df['sentence'].progress_apply(compress_gzip)

    # apply bzip2
    tqdm.pandas(desc="Applying bzip2 lossless baseline")
    df['bz2'] = df['sentence'].progress_apply(compress_bz2)

    # evaluate lossless algorithms
    df['gzip_cr'] = df.apply(
        lambda row: 1 - len(row["gzip"]) / len(row["sentence"]), axis=1
    )
    df['bz2_cr'] = df.apply(
        lambda row: 1 - len(row["bz2"])/ len(row["sentence"]) , axis=1
    )

    # print results
    print("Lossless Algorithms Compression Ratio:")
    print(f"gzip: {df['gzip_cr'].mean() * 100:.1f}%")
    print(f"bzip2: {df['bz2_cr'].mean() * 100:.1f}%")
    print()


def lossless_postprocess(df):
    """
    Only apply gzip for postprocessing evaluation
    """
    print("#### Evaluating on Lossless Postprocessing ####")
    df = df.copy()

    tqdm.pandas(desc="Applying gzip Postprocessing")
    df['gzip_postprocessing'] = df['compressed_to_short'].progress_apply(compress_gzip)

    cr_post = (1 - df['gzip_postprocessing'].apply(len) / df['sentence'].apply(len)).mean()
    print(f"CR After gzip Postprocessing: {cr_post * 100:.1f}%\n")
