import os
import platform
import subprocess
import sys


class DataDownloader:
    def __init__(self, dgl_wheel_id: str, dgl_wheel_filename: str, dvc_url: str):
        self.dgl_wheel_id = dgl_wheel_id
        self.dgl_wheel_filename = dgl_wheel_filename
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

    def download_and_install_dgl_wheel(self):
        import gdown

        print(f"Downloading DGL wheel: {self.dgl_wheel_filename}")
        gdown.download(
            id=self.dgl_wheel_id, output=self.dgl_wheel_filename, quiet=False
        )

        print(f"Installing DGL wheel: {self.dgl_wheel_filename}")
        self.run_command(f"{sys.executable} -m pip install {self.dgl_wheel_filename}")

        os.remove(self.dgl_wheel_filename)
        print(f"Removed temporary wheel file: {self.dgl_wheel_filename}")

    def download_data(self) -> None:
        self.install_gdown()
        import gdown

        self.download_and_install_dgl_wheel()

        print("Downloading dvc data...")
        gdown.download_folder(self.dvc_url, quiet=True)


def main():
    dvc = "https://drive.google.com/drive/folders/146jwbgPPPmPC2v582SxlwW_e-N2XsjEZ?usp=drive_link"
    if platform.system() == "Linux":
        downloader = DataDownloader(
            dgl_wheel_id="1cWbJLZev-yCRtVaT2klIyewNnTIOQgpF",
            dgl_wheel_filename="dgl-2.4.0-cp312-cp312-manylinux1_x86_64.whl",
            dvc_url=dvc,
        )
    elif platform.system() == "Darwin":
        downloader = DataDownloader(
            dgl_wheel_id="1f5AVz4sFkmfCnqq5Rz6FNTGW6wgMP_aD",
            dgl_wheel_filename="dgl-2.2.1-cp312-cp312-macosx_12_0_arm64.whl",
            dvc_url=dvc,
        )
    elif platform.system() == "Windows":
        downloader = DataDownloader(
            dgl_wheel_id="1fu-iLQED-FHw4FGbLkbGbL2TIIgbHEIY",
            dgl_wheel_filename="dgl-2.2.1-cp312-cp312-win_amd64.whl",
            dvc_url=dvc,
        )
    else:
        raise OSError("Unsupported operating system")

    downloader.download_data()


if __name__ == "__main__":
    main()
