from dagster import Definitions, load_assets_from_modules

from ml_pipelines.assets import iris

all_assets = load_assets_from_modules([iris])

defs = Definitions(
    assets=all_assets,
)
