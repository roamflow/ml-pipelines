import os
from dagster import Definitions, FilesystemIOManager, load_assets_from_modules

from ml_pipelines.assets import iris

all_assets = load_assets_from_modules([iris])

storage_dir = os.getenv("DAGSTER_STORAGE_DIR", "/tmp/dagster_storage")

defs = Definitions(
    assets=all_assets,
    resources={
        "io_manager": FilesystemIOManager(base_dir=storage_dir),
    },
)
