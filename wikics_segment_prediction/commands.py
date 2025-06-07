import fire
import hydra


def train(*overrides, config_path: str = "configs", config_name: str = "config"):
    from .train import run_training

    with hydra.initialize(
        version_base=None, config_path=config_path, job_name="wiki_cs_train"
    ):
        cfg = hydra.compose(config_name=config_name, overrides=list(overrides))
        run_training(cfg)


def infer(*overrides, config_path: str = "configs", config_name: str = "config"):
    from .infer import run_testing

    with hydra.initialize(
        version_base=None, config_path=config_path, job_name="wiki_cs_test"
    ):
        cfg = hydra.compose(config_name=config_name, overrides=list(overrides))
        run_testing(cfg)


def download():
    from .download_data import DataDownloader

    downloader = DataDownloader(
        "152ftcEEKftLs3WqUGKvaPo8H_gbneO53",
        "https://drive.google.com/drive/folders/146jwbgPPPmPC2v582SxlwW_e-N2XsjEZ?usp=drive_link",
    )
    downloader.download_data()


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "download": download,
        }
    )
