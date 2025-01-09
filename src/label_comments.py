import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.data.loader import load_submissions_df, load_comments_df
import src.data.constants as dataset_constants
import re

# Load data
f1_submissions_df = load_submissions_df(
    dataset_constants.RawFile.FORMULA1_SUBMISSIONS,
    columns=dataset_constants.DEFAULT_SUBMISSION_COLUMNS | {'permalink', 'post_hint', 'link_flair_text'},
).head(5)
f1_comments_df = load_comments_df(
    dataset_constants.RawFile.FORMULA1_COMMENTS,
    columns=dataset_constants.DEFAULT_COMMENT_COLUMNS | {'link_id'},
).head(100000)

# Preprocess to filter for steward decision-related posts
steward_decision_related_words = {
    'penalty', 'steward', 'decision', 'appeal', 'review', 'ruling', 'investigation', 'regulation',
    'seconds', 'sec', 'collision', 'crash', 'incident', 'overtake', 'virtual safety car', 'blocking', 'brake test',
    'contact', 'red flag', 'yellow flag', 'controversial', 'rigged', 'corrupt', 'bias', 'protest', 'FIA', 'document', 'infringement'
}

excluded_submission_ids = {
    'vdr1c6', 'w7z5aj', 'wf87e0', 'x1zd5z', 'x3y140'
}

words_regex = ''.join(fr'\b{word}\b|' for word in steward_decision_related_words)[:-1]
steward_decision_pattern = re.compile(words_regex, flags=re.IGNORECASE)
relevant_flairs = {':post-technical: Technical', ':post-news: News'}

has_related_words = f1_submissions_df['title'].apply(lambda title: steward_decision_pattern.search(title) is not None)
has_relevant_flairs = f1_submissions_df['link_flair_text'].isin(relevant_flairs)
is_image_post = f1_submissions_df['post_hint'] == 'image'
is_included = ~f1_submissions_df['id'].isin(excluded_submission_ids)

steward_decision_submissions_df = f1_submissions_df[has_related_words & has_relevant_flairs & is_image_post & is_included]

# Allow user to select a submission ID and label comments
def label_comments_for_submission(submission_id: str):
    submission_link_id = f't3_{submission_id}'
    comments_df = f1_comments_df[f1_comments_df['link_id'] == submission_link_id]

    if comments_df.empty:
        print("No comments for this submission.")
        return

    labeled_comments = []

    for idx, row in comments_df.iterrows():
        print(f"Comment {row['id']}: {row['body']}")
        label = input("Enter sentiment label (1=Negative, 2=Neutral, 3=Positive): ")
        while label not in ['1', '2', '3']:
            label = input("Invalid input. Enter sentiment label (1=Negative, 2=Neutral, 3=Positive): ")
        
        labeled_comments.append({
            "comment_id": row['id'],
            "sentiment": int(label) - 2  # Convert to -1, 0, 1 scale
        })

    # Save labeled comments to an NDJSON file
    output_file = Path(f"labeled_comments_{submission_id}.ndjson")
    with open(output_file, 'w') as f:
        for labeled_comment in labeled_comments:
            f.write(json.dumps(labeled_comment) + '\n')
    print(f"Labeled comments saved to {output_file}")

def main():
    submission_id = input("Enter submission ID: ")
    label_comments_for_submission(submission_id)

if __name__ == '__main__':
    main()
