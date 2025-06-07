import subprocess
import sys


class DataDownloader:
    def __init__(self, dgl_id: str, dvc_url: str):
        self.dgl_id = dgl_id
        self.dvc_url = dvc_url

    def run_command(self, command, capture_output=False):
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
            print(f"Error running command: {command}")
            print(f"Error details: {e.stderr}")
            sys.exit(1)

    def install_gdown(self):
        print("Installing gdown...")
        self.run_command(f"{sys.executable} -m pip install gdown")

    def download_data(self) -> None:
        self.install_gdown()
        import gdown

        self.run_command(f"gdown {self.dgl_id}")
        print("Downloading dvc data...")
        gdown.download_folder(self.dvc_url, quiet=True)
