import polars as pl
import pulp
from datasets import load_dataset
from typing_extensions import Self

from config import (
    DISK_PROCESSED_PATH,
    DISK_RAW_PATH,
    HF_PROCESSED_DATASET,
    HF_RAW_DATASET,
    MIN_REVIEWS_BUSINESS,
    MIN_REVIEWS_USER,
    RAW_BUSINESS,
    RAW_REVIEW,
    RAW_USER,
    TEST_FILE,
    TEST_REVIEWS_PER_USER,
    TRAIN_FILE,
    VALID_FILE,
    VALID_REVIEWS_PER_USER,
)


class DataPreparation:
    def __init__(
        self,
    ):
        self.reviews_df: pl.LazyFrame = pl.LazyFrame()
        self.business_df: pl.LazyFrame = pl.LazyFrame()
        self.users_df: pl.LazyFrame = pl.LazyFrame()
        self.train: pl.DataFrame = pl.DataFrame()
        self.valid: pl.DataFrame = pl.DataFrame()
        self.test: pl.DataFrame = pl.DataFrame()

    def load_raw_from_disk(
        self,
        path: str = DISK_RAW_PATH,
        review_raw_file: str = RAW_REVIEW,
        business_raw_file: str = RAW_BUSINESS,
        users_raw_file: str = RAW_USER,
    ) -> Self:
        """Load the DataFrames from json files on disk.

        Parameters
        ----------
        disk_raw_path : str
            Path to the directory where the DataFrames are saved.
        reviews_raw_file : str
            Name of the file containing the reviews DataFrame.
        business_raw_file : str
            Name of the file containing the business DataFrame.
        users_raw_file : str
            Name of the file containing the users DataFrame.

        Returns
        -------
        Self
        """
        self.reviews_df = pl.scan_ndjson(review_raw_file)
        self.business_df = pl.scan_ndjson(business_raw_file)
        self.users_df = pl.scan_ndjson(users_raw_file)
        return self

    def load_raw_from_hf(
        self,
        dataset: str = HF_RAW_DATASET,
        review_raw_file: str = RAW_REVIEW,
        business_raw_file: str = RAW_BUSINESS,
        users_raw_file: str = RAW_USER,
    ) -> Self:
        """Load the DataFrames from Hugging Face Datasets.

        Parameters
        ----------
        dataset : str
            Name of the dataset to load.
        review_raw_file : str
            Name of the file containing the reviews DataFrame.
        business_raw_file : str
            Name of the file containing the business DataFrame.
        users_raw_file : str
            Name of the file containing the users DataFrame.

        Returns
        -------
        Self
        """
        data_files = [review_raw_file, business_raw_file, users_raw_file]
        dataset = load_dataset(dataset, data_files=data_files)
        self.reviews_df = dataset[review_raw_file].to_polars().lazy()
        self.business_df = dataset[business_raw_file].to_polars().lazy()
        self.users_df = dataset[users_raw_file].to_polars().lazy()

    def filter_businesses_and_users(
        self,
        min_reviews_business: int = MIN_REVIEWS_BUSINESS,
        min_reviews_user: int = MIN_REVIEWS_USER,
    ) -> Self:
        """Filter businesses and users based on the number of reviews they have.

        Parameters
        ----------
        min_reviews_business : int
            Lower bound for the number of reviews a business should have.
        min_reviews_user : int
            Lower bound for the number of reviews a user should have.

        Returns
        -------
        Self
        """
        print("Original number of reviews:", len(self.reviews_df))
        print("Original number of businesses:", len(self.business_df))
        print("Original number of users:", len(self.users_df))

        data_per_business = (
            self.reviews_df.group_by("business_id")
            .len()
            .rename({"len": "n_reviews_business"})
            .filter(pl.col("n_reviews_business") >= min_reviews_business)
        )
        data_per_user = (
            self.reviews_df.group_by("user_id")
            .len()
            .rename({"len": "n_reviews_user"})
            .filter(pl.col("n_reviews_user") >= min_reviews_user)
        )
        self.reviews_df = self.reviews_df.join(
            data_per_business, on="business_id", how="inner"
        ).join(data_per_user, on="user_id", how="inner")
        self.business_df = self.business_df.join(
            data_per_business, on="business_id", how="inner"
        )
        self.users_df = self.users_df.join(data_per_user, on="user_id", how="inner")

        print("Number of reviews after filtering:", len(self.reviews_df))
        print("Number of businesses after filtering:", len(self.business_df))
        print("Number of users after filtering:", len(self.users_df))

        return self

    def minimize_n_reviews(
        self,
        min_reviews_business: int = MIN_REVIEWS_BUSINESS,
        min_reviews_user: int = MIN_REVIEWS_USER,
    ) -> Self:
        """Minimize the number of reviews such that most businesses and users have at least the specified number of reviews.
        Prioritizes longer (N_chars) reviews.

        Parameters
        ----------
        min_reviews_business : int
            Number of reviews per business to keep.
        min_reviews_user : int
            Number of reviews per user to keep.

        Returns
        -------
        Self
        """
        print("Original number of reviews:", len(self.reviews_df))

        # Add an index to the DataFrame and normalize the number of characters in the reviews
        reviews_df = self.reviews_df.with_columns(
            index=pl.arange(0, len(self.reviews_df)),
            n_chars=pl.col("text").str.len_chars(),
        ).with_columns(
            normalized_n_chars=pl.col("n_chars")
            - pl.col("n_chars").min() / pl.col("n_chars").max()
            - pl.col("n_chars").min()
        )

        # Define the problem
        prob = pulp.LpProblem("Minimize_Reviews", pulp.LpMinimize)

        # Number of reviews
        n_reviews = len(reviews_df)

        # Define the decision variables
        x = pulp.LpVariable.dicts("review", range(n_reviews), cat=pulp.LpBinary)

        # Objective function: Minimize the number of reviews, prefer longer texts
        text_lengths = reviews_df["n_chars"]

        prob += (
            pulp.lpSum(x[i] * (-text_lengths[i]) for i in range(n_reviews)),
            "Total reviews",
        )

        # Constraint: All users in the sample should have at least N_REVIEWS_USER
        for user_id in reviews_df["user_id"].unique():
            user_reviews = reviews_df.filter(pl.col("user_id") == user_id)
            prob += (
                pulp.lpSum(x[i] for i in user_reviews["index"]) >= min_reviews_user,
                f"User_{user_id}_constraint",
            )

        # Constraint: All businesses in the sample should have at least N_REVIEWS_BUSINESS
        for business_id in reviews_df["business_id"].unique():
            business_reviews = reviews_df.filter(pl.col("business_id") == business_id)
            prob += (
                pulp.lpSum(x[i] for i in business_reviews["index"])
                >= n_reviews_business,
                f"Business_{business_id}_constraint",
            )

        # Solve the problem
        prob.solve()

        reviews_df = reviews_df.with_columns(
            pl.Series("keep", [x[i].varValue for i in range(n_reviews)])
        )
        self.reviews_df = reviews_df.filter(pl.col("keep") == 1).select(
            self.reviews_df.columns
        )
        print("Number of reviews after minimizing:", len(self.reviews_df))
        return self

    def split_train_valid_test(
        self,
        valid_reviews_per_user: int = VALID_REVIEWS_PER_USER,
        test_reviews_per_user: int = TEST_REVIEWS_PER_USER,
    ) -> Self:
        """Split the reviews DataFrame into train, validation, and test sets according to the user_fraction.

        Parameters
        ----------
        valid_reviews_per_user : float
            Fraction of the data to use for validation.
        test_user_fraction : float
            Fraction of the data to use for testing.

        Returns
        -------
        Self
        """

    def save_processed_to_hf(
        self,
        dataset: str = HF_PROCESSED_DATASET,
        train_file: str = TRAIN_FILE,
        valid_file: str = VALID_FILE,
        test_file: str = TEST_FILE,
    ) -> Self:
        pass

    def save_processed_to_disk(
        self,
        disk_path: str = DISK_PROCESSED_PATH,
        train_file: str = TRAIN_FILE,
        valid_file: str = VALID_FILE,
        test_file: str = TEST_FILE,
    ) -> Self:
        """Save the filtered DataFrames to parquet files.

        Parameters
        ----------
        reviews_file : str
            Path to save the filtered reviews DataFrame.
        business_file : str
            Path to save the filtered business DataFrame.
        users_file : str
            Path to save the filtered users DataFrame.
        """
        reviews_df = self.reviews_df.collect()
        business_df = self.business_df.collect()
        users_df = self.users_df.collect()
        reviews_df.write_parquet(f"{disk_path}/{train_file}")
        business_df.write_parquet(f"{disk_path}/{valid_file}")
        users_df.write_parquet(f"{disk_path}/{test_file}")

    def load_processed_from_disk(
        self,
        disk_path: str = DISK_PROCESSED_PATH,
        train_file: str = TRAIN_FILE,
        valid_file: str = VALID_FILE,
        test_file: str = TEST_FILE,
    ) -> Self:
        """Load the filtered DataFrames from parquet files.

        Parameters
        ----------
        disk_path : str
            Path to the directory where the DataFrames are saved.
        train_file : str
            Name of the file containing the training DataFrame.
        valid_file : str
            Name of the file containing the validation DataFrame.
        test_file : str
            Name of the file containing the testing DataFrame.

        Returns
        -------
        Self
        """
        self.train = pl.read_parquet(f"{disk_path}/{train_file}")
        self.valid = pl.read_parquet(f"{disk_path}/{valid_file}")
        self.test = pl.read_parquet(f"{disk_path}/{test_file}")
        return self

    def load_processed_from_hf(
        self,
        dataset: str = HF_PROCESSED_DATASET,
        train_file: str = TRAIN_FILE,
        valid_file: str = VALID_FILE,
        test_file: str = TEST_FILE,
    ) -> Self:
        """Load the filtered DataFrames from Hugging Face Datasets.

        Parameters
        ----------
        dataset : str
            Name of the dataset to load.
        train_file : str
            Name of the file containing the training DataFrame.
        valid_file : str
            Name of the file containing the validation DataFrame.
        test_file : str
            Name of the file containing the testing DataFrame.

        Returns
        -------
        Self
        """
        data_files = [train_file, valid_file, test_file]
        dataset = load_dataset(dataset, data_files=data_files)
        self.train = dataset[train_file].to_polars()
        self.valid = dataset[valid_file].to_polars()
        self.test = dataset[test_file].to_polars()
        return self
