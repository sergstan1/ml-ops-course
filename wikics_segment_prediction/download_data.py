import subprocess

import gdown


class DataDownloader:
    def __init__(self, dgl_id: str, dvc_url: str):
        self.dgl_id = dgl_id
        self.dvc_url = dvc_url

    def download_data(self) -> None:
        subprocess.run(
            ["poetry", "run", "gdown", self.dgl_id], capture_output=True, text=True
        )
        gdown.download_folder(self.dvc_url, quiet=True)


if __name__ == "__main__":
    downloader = DataDownloader(
        "152ftcEEKftLs3WqUGKvaPo8H_gbneO53",
        "https://drive.google.com/drive/folders/177T5k6paCCu9td9zibOAIJ2b803neRF_?usp=drive_link",
    )
    downloader.download_data()
