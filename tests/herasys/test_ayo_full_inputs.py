from benchmarks.herasys import samples
from scripts.reproduce_ayo_full import samples as native_samples


def test_native_ayo_and_shared_backends_receive_identical_inputs():
    records = [
        {"question": f"question {i}", "documents": [f"a {i}", f"b {i}"]}
        for i in range(50)
    ]
    for count in (1, 12, 60, 1000):
        for seed in (42, 43, 44):
            assert native_samples(records, count, seed, 0.1) == samples(
                records, count, seed, 0.1
            )
