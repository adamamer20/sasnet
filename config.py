#### Data Preparation ####
HF_RAW_DATASET = "adamamer20/yelp-04-2024"
HF_RAW_PATH = "yelp-dataset"
DISK_RAW_PATH = "data/raw/yelp-dataset"
RAW_REVIEW = "yelp_academic_dataset_review.json"
RAW_BUSINESS = "yelp_academic_dataset_business.json"
RAW_USER = "yelp_academic_dataset_user.json"

HF_PROCESSED_DATASET = "adamamer20/sasnet-yelp"
DISK_PROCESSED_PATH = "data/processed/yelp-dataset"
TRAIN_FILE = "train.parquet"
VALID_FILE = "valid.parquet"
TEST_FILE = "test.parquet"

MIN_REVIEWS_BUSINESS = 15
MIN_REVIEWS_USER = 15

VALID_REVIEWS_PER_USER = 3
TEST_REVIEWS_PER_USER = 3

#### Embeddings creation ####
