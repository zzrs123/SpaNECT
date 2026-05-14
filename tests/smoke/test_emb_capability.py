"""Smoke test for the embedding capability package."""

from spanect import SpaNECT as sn
from spanect.emb.vision_embed import VisionEmbed as EmbVisionEmbed


def main():
    checks = {
        "sn.emb": hasattr(sn, "emb"),
        "sn.emb.VisionEmbed": hasattr(sn.emb, "VisionEmbed"),
        "sn.emb.vision_embed": hasattr(sn.emb, "VisionEmbed"),
    }
    for key, ok in checks.items():
        print(f"{key}: {ok}")

    if not all(checks.values()):
        raise SystemExit(1)

    assert sn.emb.VisionEmbed is EmbVisionEmbed, "sn.emb should expose emb VisionEmbed"

    print("embedding capability smoke test passed")


if __name__ == "__main__":
    main()
