import pandas as pd
import joblib
import os

class ExportManager:
    def __init__(self, export_dir="exports"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def export_csv(self, df, filename="cleaned_data.csv"):
        path = os.path.join(self.export_dir, filename)
        df.to_csv(path, index=False)
        return path

    def export_model(self, model, filename="trained_model.pkl"):
        path = os.path.join(self.export_dir, filename)
        joblib.dump(model, path)
        return path

    def export_plot(self, fig, filename="plot.png"):
        path = os.path.join(self.export_dir, filename)
        fig.savefig(path)
        return path
