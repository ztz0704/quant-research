import pytest

from financial_ml.data import synthetic
from financial_ml.labels import add_labels
from financial_ml.temporal import partitions, walk_forward


def labeled():
    frame = add_labels(synthetic(days=90, assets=4), 5)
    frame["eligible"] = True
    return frame


@pytest.mark.parametrize("mode", ["rolling", "expanding"])
def test_date_blocks_and_purging(mode):
    frame = labeled()
    settings = dict(mode=mode, train_days=20, validation_days=10, test_days=10, step_days=10, embargo_days=2)
    folds = walk_forward(frame, settings)
    used = set()
    for fold in folds:
        train, val, test = partitions(frame, fold)
        assert train.date.max() < val.date.min() < test.date.min()
        assert train.label_end.max() < fold.validation_start
        assert val.label_end.max() < fold.test_start
        assert not used.intersection(test.date.unique())
        used.update(test.date.unique())
        assert test.groupby("date").size().eq(4).all()
    if mode == "expanding":
        assert folds[-1].train_start == folds[0].train_start
    else:
        assert folds[-1].train_start > folds[0].train_start


def test_future_target_status_not_used_for_test_composition():
    frame = labeled()
    settings = dict(mode="rolling", train_days=20, validation_days=10, test_days=10, step_days=10, embargo_days=2)
    fold = walk_forward(frame, settings)[0]
    before = partitions(frame, fold)[2]
    frame.loc[frame.date >= fold.test_start, "target_valid"] = False
    after = partitions(frame, fold)[2]
    assert before.index.equals(after.index)


def test_insufficient_and_overlap_rejected():
    frame = labeled()
    settings = dict(mode="rolling", train_days=100, validation_days=10, test_days=10, step_days=10, embargo_days=2)
    with pytest.raises(ValueError):
        walk_forward(frame, settings)
    settings.update(train_days=20, step_days=5)
    with pytest.raises(ValueError):
        walk_forward(frame, settings)
