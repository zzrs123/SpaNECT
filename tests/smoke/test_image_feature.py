"""
Smoke test for image feature acquisition module.

Tests the new tl._image_feat module.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))


def test_get_image_feat_from_h5ad():
    """Test getting image feature from h5ad obsm."""
    from types import SimpleNamespace
    from spanect.tl._image_feat import get_image_feat

    fake_embeddings = np.random.randn(10, 32).astype(np.float32)
    adata = SimpleNamespace(obsm={'image_feat': fake_embeddings})

    result = get_image_feat(
        adata=adata,
        data_path="/tmp",
        data_name="test",
        use_h5ad=True,
    )

    assert result.shape == (10, 32)
    np.testing.assert_array_equal(result, fake_embeddings)
    print("[PASS] test_get_image_feat_from_h5ad")


def test_get_image_feat_from_npy():
    """Test getting image feature from embeddings.npy file."""
    from types import SimpleNamespace
    from spanect.tl._image_feat import get_image_feat

    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = tmpdir
        data_name = "test_slice"
        data_dir = Path(data_path) / data_name
        data_dir.mkdir(parents=True, exist_ok=True)

        fake_embeddings = np.random.randn(10, 32).astype(np.float32)
        emb_path = data_dir / "embeddings.npy"
        np.save(emb_path, fake_embeddings)

        adata = SimpleNamespace(obsm={})

        result = get_image_feat(
            adata=adata,
            data_path=data_path,
            data_name=data_name,
            use_h5ad=False,
        )

        assert result.shape == (10, 32)
        np.testing.assert_array_almost_equal(result, fake_embeddings)
        assert 'image_feat' in adata.obsm
        print("[PASS] test_get_image_feat_from_npy")


def test_priority_h5ad_obsm_over_npy():
    """Test that h5ad obsm takes priority over npy file."""
    from types import SimpleNamespace
    from spanect.tl._image_feat import get_image_feat

    obsm_embeddings = np.ones((10, 32), dtype=np.float32)
    npy_embeddings = np.zeros((10, 32), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "test"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.save(data_dir / "embeddings.npy", npy_embeddings)

        adata = SimpleNamespace(obsm={'image_feat': obsm_embeddings})

        result = get_image_feat(
            adata=adata,
            data_path=tmpdir,
            data_name="test",
            use_h5ad=True,
        )

        assert result.shape == (10, 32)
        np.testing.assert_array_equal(result, obsm_embeddings)
        print("[PASS] test_priority_h5ad_obsm_over_npy")


def test_generate_via_byol():
    """Test generating image feature via BYOL (mocked)."""
    from types import SimpleNamespace
    from unittest.mock import patch, MagicMock

    fake_embeddings = np.ones((10, 32), dtype=np.float32)
    adata = SimpleNamespace(obsm={})

    def mock_run(self):
        adata.obsm['image_feat'] = fake_embeddings
        return (fake_embeddings, adata)

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('spanect.emb.vision_embed.VisionEmbed.run', mock_run):
            from spanect.tl._image_feat import get_image_feat
            result = get_image_feat(
                adata=adata,
                data_path=tmpdir,
                data_name="test",
                use_h5ad=False,
                config={'img_epoch': 1},
                device='cpu',
            )

    assert result.shape == (10, 32)
    assert 'image_feat' in adata.obsm
    print("[PASS] test_generate_via_byol")


def main():
    print("=" * 60)
    print("Image Feature Acquisition Module Smoke Test")
    print("=" * 60)

    test_get_image_feat_from_h5ad()
    test_get_image_feat_from_npy()
    test_priority_h5ad_obsm_over_npy()
    test_generate_via_byol()

    print("\n[PASS] All image feature tests passed!")


if __name__ == "__main__":
    main()
