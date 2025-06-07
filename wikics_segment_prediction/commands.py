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


if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "infer": infer,
        }
    )
