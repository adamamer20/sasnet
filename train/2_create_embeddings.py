import polars as pl


class EmbeddingsCreator:
    def __init__(
        self, train_df: pl.DataFrame, valid_df: pl.DataFrame, test_df: pl.DataFrame
    ):
        self.train_df : pl.DataFrame = train_df
        self.valid_df : pl.DataFrame= valid_df
        self.test_df : pl.DataFrame = test_df  
    
