'''Constants related to the subreddit datasets'''

import numpy as np
import datetime as dt
from src.utils import infer_types
from config import config

YEAR = 2022
START_DATE = dt.datetime(2022, 6, 1)
END_DATE = dt.datetime(2023, 1, 1) - dt.timedelta(microseconds=1)

class RawFile:
    FORMULA1_SUBMISSIONS = config.RAW_DATA_DIR / 'formula1_submissions.ndjson'
    FORMULA1_COMMENTS = config.RAW_DATA_DIR / 'formula1_comments.ndjson'
    FORMULA1POINT5_SUBMISSIONS = config.RAW_DATA_DIR / 'formula1point5_submissions.ndjson'
    FORMULA1POINT5_COMMENTS = config.RAW_DATA_DIR / 'formula1point5_comments.ndjson'

SUBMISSION_COLUMN_DTYPES = infer_types({
    # 'all_awardings',
    # 'allow_live_comments',
    # 'archived',

    # The account name of the poster, e.g., “example username” (String)
    'author': str,

    # 'author_created_utc',
    # 'author_flair_background_color',
    # 'author_flair_css_class',
    # 'author_flair_richtext',
    # 'author_flair_template_id',
    # 'author_flair_text',
    # 'author_flair_text_color',
    # 'author_flair_type',
    # 'author_fullname',
    # 'author_patreon_flair',
    # 'author_premium',
    # 'awarders',
    # 'banned_by',
    # 'can_gild',
    # 'can_mod_post',
    # 'category',
    # 'content_categories',
    # 'contest_mode',
    
    # UNIX timestamp referring to the time of the submission’s creation, e.g., 1483228803 (Integer).
    'created_utc': np.dtype('datetime64[s]'),

    # 'discussion_type',
    # 'distinguished',
    # 'domain',
    # 'edited',

    'gilded': np.uint8, # Value range in dataset is 0 through 12

    # 'gildings',
    # 'hidden',
    # 'hide_score',

    # The submission’s identifier, e.g., “5lcgjh” (String).
    'id': str,

    # 'is_created_from_ads_ui',
    # 'is_crosspostable',
    # 'is_meta',
    # 'is_original_content',
    # 'is_reddit_media_domain',
    # 'is_robot_indexable',
    # 'is_self',
    # 'is_video',
    # 'link_flair_background_color',
    # 'link_flair_css_class',
    # 'link_flair_richtext',
    # 'link_flair_template_id',
    # 'link_flair_text',
    # 'link_flair_text_color',
    # 'link_flair_type',
    # 'locked',
    # 'media',
    # 'media_embed',
    # 'media_only',
    # 'name',
    # 'no_follow',
    # 'num_comments',
    # 'num_crossposts',
    # 'over_18',
    # 'parent_whitelist_status',
    # 'permalink',
    # 'pinned',
    # 'pwls',
    # 'quarantine',
    # 'removed_by',
    # 'removed_by_category',
    # 'retrieved_on',
    # 'retrieved_utc',
    
    # The score that the submission has accumulated. The score is the number of upvotes minus
    # the number of downvotes. E.g., 5 (Integer). NB: Reddit fuzzes the real score to prevent spam bots.
    'score': np.int32, # Value range in dataset is -322 through 62910

    # 'secure_media',
    # 'secure_media_embed',
    
    # The text that is associated with the submission (String).
    'selftext': str,

    # 'send_replies',
    # 'spoiler',
    # 'stickied',
    # 'subreddit',
    # 'subreddit_id',
    # 'subreddit_name_prefixed',
    # 'subreddit_subscribers',
    # 'subreddit_type',
    # 'suggested_sort',
    # 'thumbnail',
    # 'thumbnail_height',
    # 'thumbnail_width',
    
    # The title that is associated with the submission, e.g., “What did you think of the ending of Rogue One?” (String)
    'title': str,

    # 'top_awarded_type',
    # 'total_awards_received',
    # 'treatment_tags',
    # 'upvote_ratio', # TODO: is this interesting?
    # 'url',
    # 'url_overridden_by_dest',
    # 'view_count',
    # 'whitelist_status',
    # 'wls',
})
SUBMISSION_COLUMNS = frozenset(SUBMISSION_COLUMN_DTYPES.keys())

COMMENT_COLUMN_DTYPES = infer_types({
    # 'all_awardings',
    # 'archived',
    # 'associated_award',
    
    # The account name of the poster, e.g., “example username” (String).
    'author': str,

    # 'author_created_utc',
    # 'author_flair_background_color',
    # 'author_flair_css_class',
    # 'author_flair_richtext',
    # 'author_flair_template_id',
    # 'author_flair_text',
    # 'author_flair_text_color',
    # 'author_flair_type',
    # 'author_fullname',
    # 'author_patreon_flair',
    # 'author_premium',

    # The comment’s text, e.g., “This is an example comment” (String).
    'body': str,
    
    # 'can_gild',
    # 'collapsed',
    # 'collapsed_because_crowd_control',
    # 'collapsed_reason',
    # 'collapsed_reason_code',
    # 'comment_type',
    # 'controversiality', # TODO:

    # UNIX timestamp that refers to the time of the submission’s creation, e.g., 1483228803 (Integer).
    'created_utc': np.dtype('datetime64[s]'),

    # 'distinguished',
    # 'edited',
    
    # The number of times this comment received Reddit gold, e.g., 0 (Integer).
    'gilded': np.uint8, # Value range in dataset is 0 through 12
    
    # 'gildings',
    
    # The comment’s identifier, e.g., “dbumnq8” (String).
    'id': str,
    
    # 'is_submitter',
    # 'link_id',
    # 'locked',
    # 'name',
    # 'no_follow',
    # 'parent_id',
    # 'permalink',
    # 'retrieved_on',
    
    # The score of the comment. The score is the number of upvotes minus
    # the number of downvotes. Note that Reddit fuzzes the real score to prevent spam bots. E.g., 5 (Integer).
    'score': np.int32, # Value range in dataset is -322 through 62910
    
    # 'score_hidden',
    # 'send_replies',
    # 'stickied',
    # 'subreddit',
    # 'subreddit_id',
    # 'subreddit_name_prefixed',
    # 'subreddit_type',
    # 'top_awarded_type',
    # 'total_awards_received',
    # 'treatment_tags',
    # 'unrepliable_reason',
})
COMMENT_COLUMNS = frozenset(COMMENT_COLUMN_DTYPES.keys())
