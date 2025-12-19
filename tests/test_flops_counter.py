import unittest
import torch


from nncore.flops.flops_counter import get_model_complexity_info_general
from nncore.models import TinyTransformer


class TestFlopsCounter(unittest.TestCase):
    def test_get_model_complexity_info_general(self) -> None:
        model = TinyTransformer(vocab_size=10000)
        input = torch.randint(0, 10000, (2, 64))  # 批量2，长度64
        flops_count, params_count = get_model_complexity_info_general(
            model, args=(input,), tgt_module=model, verbose=False
        )
        self.assertIsNotNone(flops_count)
        self.assertIsNotNone(params_count)
        print(f"FLOPs: {flops_count}, Params: {params_count}")
