import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionConfig,
    Qwen3_5MoeVisionModel,
)


class Qwen35VisionMetadataTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        attention_backend = patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe."
            "qwen3_5_moe_vit.default_attn_impl",
            "sdpa",
        )
        attention_backend.start()
        self.addCleanup(attention_backend.stop)
        self.config = Qwen3_5MoeVisionConfig(
            depth=2,
            hidden_size=32,
            intermediate_size=48,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            out_hidden_size=16,
            num_position_embeddings=16,
        )
        self.config._attn_implementation = "sdpa"

    @torch.inference_mode()
    def test_host_lengths_match_attention_without_reading_tensor_values(self):
        attention = Qwen3_5MoeVisionAttention(self.config).eval()
        hidden = torch.randn(28, 32)
        positions = (torch.randn(28, 8), torch.randn(28, 8))
        cumulative = torch.tensor([0, 8, 16, 28], dtype=torch.int32)
        expected = attention(hidden, cumulative, position_embeddings=positions)
        with patch.object(
            torch.Tensor, "tolist", side_effect=AssertionError("readback")
        ):
            actual = attention(
                hidden,
                cumulative,
                position_embeddings=positions,
                sequence_lengths=(8, 8, 12),
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_batched_video_frames_preserve_attention_boundaries(self):
        model = Qwen3_5MoeVisionModel(self.config).eval()
        grids = torch.tensor([[2, 2, 4], [1, 4, 2]])
        pixels = torch.randn(24, 24)
        actual = model(pixels, grid_thw=grids)
        # Each temporal frame is an independent attention sequence. Running
        # frames separately must preserve ordering and both output tensors.
        outputs = [
            model(pixels[index * 8 : (index + 1) * 8], grid_thw=grid)
            for index, grid in enumerate(
                [torch.tensor([[1, 2, 4]]), torch.tensor([[1, 2, 4]]), grids[1:]]
            )
        ]
        for name in ("last_hidden_state", "pooler_output"):
            expected = torch.cat([getattr(output, name) for output in outputs])
            torch.testing.assert_close(getattr(actual, name), expected)

    def test_position_ids_allow_separate_metadata_and_output_devices(self):
        embedding = SimpleNamespace(visual=SimpleNamespace(spatial_merge_size=2))
        grids = torch.tensor([[2, 4, 2], [0, 4, 2]], dtype=torch.int64)
        outputs = Qwen3_5MoeImageEmbedding.get_position_ids(
            embedding, grids, device=torch.device("cpu")
        )
        torch.testing.assert_close(
            outputs[0],
            torch.tensor(
                [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=torch.int32
            ),
        )
        self.assertEqual(outputs[1].shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
