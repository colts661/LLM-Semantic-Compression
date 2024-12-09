import json
import openai
import backoff
from tqdm import tqdm

MODEL = "gpt-4o-mini"

def load_prompt(prompt, text_genre='document'):
    """
    Loading prompts from file
    """
    with open(f'./prompts/{prompt}.json', encoding='UTF-8') as f:
        full_prompt = json.load(f)

        if 'system' not in full_prompt:
            full_prompt['system'] = 'You are a helpful assistant'

        return {
            'system': full_prompt['system'].replace('[TEXT_GENRE]', text_genre),
            'action': full_prompt['action'].replace('[TEXT_GENRE]', text_genre)
        }


def load_classification_prompt(text_genre, task="classify-topic"):
    """
    Loading prompts for text classification from file
    """
    with open(f'./prompts/{task}.json', encoding='UTF-8') as f:
        full_prompt = json.load(f)
        return {
            'system': full_prompt['system'],
            'action': full_prompt['action'].replace('[TEXT_GENRE]', text_genre)
        }


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def generate_text(document, client, prompt):
    """
    Make OpenAI API Calls while maintaining API call rate limit 
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": prompt['action'].replace('[TEXT]', document)}
        ]
    )
    return response.choices[0].message.content


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def generate_text_classification(document, client, prompt, classes):
    """
    Make OpenAI API Calls on text classification while maintaining API call rate limit 
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": (
                prompt['action']
                .replace('[TEXT]', document)
                .replace('[POTENTIAL_CLASSES]', ', '.join(classes))
            )}
        ]
    )
    return response.choices[0].message.content


def run_classification(client, classify_prompt, df, label_col):
    """
    Helper: Run classification on 3 stages of compression
    """
    # classify original
    tqdm.pandas(desc=f"Classify Original")
    classification_stage_0 = df['sentence'].progress_apply(
        generate_text_classification,
        args=(client, classify_prompt, df[label_col].unique())
    )
    stage_0_acc = (classification_stage_0 == df[label_col]).mean()

    # classify after stage 1
    tqdm.pandas(desc=f"Classify After Stage 1")
    classification_stage_1 = df['compressed_to_formatted'].progress_apply(
        generate_text_classification,
        args=(client, classify_prompt, df[label_col].unique())
    )
    stage_1_acc = (classification_stage_1 == df[label_col]).mean()

    # classify after stage 2
    tqdm.pandas(desc=f"Classify After Stage 2")
    classification_stage_2 = df['compressed_to_short'].progress_apply(
        generate_text_classification,
        args=(client, classify_prompt, df[label_col].unique())
    )
    stage_2_acc = (classification_stage_2 == df[label_col]).mean()

    # report results
    print(f"{label_col} Classification Accuracy:")
    print(f"Original: {round(stage_0_acc, 3)}%")
    print(f"After Stage 1: {round(stage_1_acc, 3)}%")
    print(f"After Stage 2: {round(stage_2_acc.item(), 3)}%\n")
