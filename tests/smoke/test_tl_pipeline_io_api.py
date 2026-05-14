"""Smoke test for training pipeline-io API exposure."""

from spanect import SpaNECT as sn
from spanect.tl._pipeline_io import get_input as tl_pipeline_io_get_input


def main():
    checks = {
        "sn.tl._pipeline_io": hasattr(sn.tl, "_pipeline_io"),
        "sn.tl.get_input": hasattr(sn.tl, "get_input"),
    }
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if not all(checks.values()):
        raise SystemExit(1)

    assert sn.tl.get_input is tl_pipeline_io_get_input, "sn.tl.get_input should expose tl._pipeline_io"

    print("training pipeline-io API smoke test passed")


if __name__ == "__main__":
    main()
