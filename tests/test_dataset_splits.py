from image_to_3d_shapenet.dataset.splits import create_splits


def test_splits_are_disjoint_and_have_requested_ratio() -> None:
    splits = create_splits([f"model-{index:02d}" for index in range(20)], seed=42)
    train = set(splits["train"])
    validation = set(splits["validation"])
    test = set(splits["test"])
    assert (len(train), len(validation), len(test)) == (14, 3, 3)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)


def test_splits_are_reproducible_and_limited() -> None:
    ids = [f"model-{index:02d}" for index in range(20)]
    first = create_splits(ids, seed=42, max_train=3, max_val=2, max_test=1)
    second = create_splits(list(reversed(ids)), seed=42, max_train=3, max_val=2, max_test=1)
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "train": 3,
        "validation": 2,
        "test": 1,
    }

