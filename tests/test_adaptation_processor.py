import numpy as np

from vote2world.processors.adaptation_processor import (
    DummyVisualTokenizer,
    GenerationOnlyAdaptationProcessor,
    ProcessorConfig,
)


def make_processor():
    ranges = np.stack([np.full(13, -1.0), np.full(13, 1.0)], axis=1).astype(np.float32)
    return GenerationOnlyAdaptationProcessor(DummyVisualTokenizer(), ranges, ProcessorConfig())


def test_generation_input_length_and_bos():
    processor = make_processor()
    batch = {
        "context_frames": np.zeros((1, 4, 3, 8, 10), dtype=np.float32),
        "context_actions": np.zeros((1, 4, 13), dtype=np.float32),
        "sample_id": ["ep0:0"],
    }
    out = processor(batch)
    assert out["gen_input_ids"].shape == (1, 1333)
    assert out["gen_input_ids"][0, 1332] == 4631
    assert out["metadata"]["history_length"] == 1332
    assert out["metadata"]["gen_output_length"] == 321


def test_action_tokens_are_offset():
    processor = make_processor()
    actions = np.zeros((1, 4, 13), dtype=np.float32)
    out = processor({"context_frames": np.zeros((1, 4, 3, 8, 10), dtype=np.float32), "context_actions": actions})
    first_action_tokens = out["gen_input_ids"][0, 320:333]
    assert first_action_tokens.min() >= 4375
    assert first_action_tokens.max() <= 4630


def test_dummy_decoder_shape():
    decoded = DummyVisualTokenizer().decode(np.zeros((1, 320), dtype=np.int64))
    assert decoded.shape == (1, 3, 256, 320)

