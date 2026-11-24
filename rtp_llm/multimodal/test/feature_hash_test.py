import unittest

import torch

from rtp_llm.ops import get_multimodal_feature_hash


class FeatureHashTest(unittest.TestCase):
    def test_cpu_output_and_content_identity(self):
        embedding = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 3.0]])
        hashes = get_multimodal_feature_hash(embedding)
        self.assertEqual(hashes.device.type, "cpu")
        self.assertEqual(hashes.dtype, torch.int32)
        self.assertEqual(hashes.shape, (3,))
        self.assertEqual(hashes[0], hashes[1])
        self.assertNotEqual(hashes[0], hashes[2])

    def test_empty_rows_rejected(self):
        for shape in ((0, 4), (3, 0)):
            with self.subTest(shape=shape), self.assertRaises(RuntimeError):
                get_multimodal_feature_hash(torch.empty(shape))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required")
    def test_cpu_gpu_bitwise_consistency(self):
        for dtype in (torch.uint8, torch.float16, torch.bfloat16, torch.float32):
            for shape in ((3, 13), (7, 31), (553, 4096)):
                with self.subTest(dtype=dtype, shape=shape):
                    cpu = torch.arange(shape[0] * shape[1]).reshape(shape).to(dtype)
                    self.assertTrue(
                        torch.equal(
                            get_multimodal_feature_hash(cpu),
                            get_multimodal_feature_hash(cpu.cuda()),
                        )
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "GPU required")
    def test_noncontiguous_input_on_nondefault_stream(self):
        cpu = torch.arange(8192, dtype=torch.float32).reshape(64, 128).t()
        expected = get_multimodal_feature_hash(cpu)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            gpu = cpu.cuda()
            self.assertFalse(gpu.is_contiguous())
            actual = get_multimodal_feature_hash(gpu)
        # No explicit synchronization: returned CPU hashes must already be ready.
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
