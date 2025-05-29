import subprocess
import sys


class DataDownloader:
    def __init__(self, dgl_id: str, dvc_url: str):
        self.dgl_id = dgl_id
        self.dvc_url = dvc_url

    def run_command(self, command, capture_output=False):
        """Run a shell command and return its output."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                text=True,
                capture_output=capture_output,
            )
            return result.stdout if capture_output else None
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running command: {command}")
            print(f"Error details: {e.stderr}")
            sys.exit(1)

    def install_gdown(self):
        """Install gdown."""
        print("⏳ Installing gdown...")
        self.run_command(f"{sys.executable} -m pip install gdown")

    def download_data(self) -> None:
        self.install_gdown()
        self.run_command(f"gdown {self.dgl_id}")
        self.run_command(f"gdown {self.dvc_url}")


def main():
    downloader = DataDownloader(
        "152ftcEEKftLs3WqUGKvaPo8H_gbneO53",
        "https://drive.google.com/drive/folders/177T5k6paCCu9td9zibOAIJ2b803neRF_?usp=drive_link",
    )
    downloader.download_data()


if __name__ == "__main__":
    main()
