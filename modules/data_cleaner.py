import pandas as pd

class DataCleaner:
    def clean(self, df, drop_duplicates=True, fill_missing=True, convert_types=True):
        original_shape = df.shape

        if drop_duplicates:
            df = df.drop_duplicates()

        if fill_missing:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna("Unknown")
                else:
                    df[col] = df[col].fillna(df[col].mean())

        if convert_types:
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
        cleaned_shape = df.shape

        return df, original_shape, cleaned_shape
