import os
import subprocess
import sys


def main() -> None:
    """Launch the Dagster gRPC server for ml-pipelines."""
    host = os.getenv("DAGSTER_GRPC_HOST", "0.0.0.0")
    port = os.getenv("DAGSTER_GRPC_PORT", "4000")
    module = os.getenv("DAGSTER_GRPC_MODULE", "ml_pipelines.definitions")

    cmd = [
        "dagster",
        "api",
        "grpc",
        "-h",
        host,
        "-p",
        str(port),
        "-m",
        module,
    ]
    sys.exit(subprocess.call(cmd))
