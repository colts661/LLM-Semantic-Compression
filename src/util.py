import matplotlib.pyplot as plt


#### Plotting Functions
def plot_stats(df):
    """
    Plots the basic statistics of a corpus dataframe
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.3, wspace=0.3)
    df['sentence'].apply(len).hist(bins=15, ax=ax[0])
    ax[0].set_title("Sentence Length Distribution")
    
    df['label'].value_counts(normalize=True).plot(kind='barh', ax=ax[1])
    ax[1].set_title("Label Distribution")

    return fig


def plot_text_length_distributions(dataset_name, df):
    """
    After multi-stage compression pipeline, plot the distributions of text length after
    each stage
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), facecolor='w', edgecolor='k')
    df['sentence'].apply(len).hist(bins=15, alpha=0.75, label='original', ax=ax)
    df['compressed_to_formatted'].apply(len).hist(bins=15, alpha=0.75, label='After 1', ax=ax)
    df['compressed_to_short'].apply(len).hist(bins=15, alpha=0.75, label='After 2', ax=ax)
    ax.legend()
    ax.set_title(f"{dataset_name} Document Length Distribution")
    return fig
