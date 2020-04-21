import pytest
import random

from milvus import ParamError

dim = 128


class TestInsert:
    vectors = [[random.random() for _ in range(dim)] for _ in range(10000)]

    def test_insert_normal(self, gcon, gcollection):
        try:
            ids = gcon.insert(gcollection, self.vectors)
            assert len(ids) == len(self.vectors)
        except Exception:
            pytest.fail("Fail raise")

    def test_insert_with_ids(self, gcon, gcollection):
        ids = [i for i in range(10000)]
        try:
            ids_ = gcon.insert(gcollection, self.vectors, ids)
            assert len(ids_) == len(self.vectors)
            assert ids == ids_
        except Exception:
            pytest.fail("Fail raise")

    def test_insert_with_partition(self, gcon, gcollection):
        gcon.create_partition(gcollection, "tag01")
        ids = gcon.insert(gcollection, self.vectors, partition_tag="tag01")
        assert len(ids) == len(self.vectors)

    def test_insert_async(self, gcon, gcollection):
        future = gcon.insert(gcollection, self.vectors, _async=True)
        ids = future.result()
        assert len(ids) == len(self.vectors)

    def test_insert_async_callback(self, gcon, gcollection):
        def cb(ids):
            assert len(ids) == len(self.vectors)

        future = gcon.insert(gcollection, self.vectors, _async=True, _callback=cb)
        future.result()
        future.done()

    @pytest.mark.parametrize("vectors", [[], None, "", 12344, [1111], [[]], [[], [1.0, 2.0]]])
    def test_insert_invalid_vectors(self, vectors, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.insert(gcollection, vectors)

    @pytest.mark.parametrize("ids", [(), "abc", [], [1, 2], [[]]])
    def test_insert_invalid_ids(self, ids, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.insert(gcollection, self.vectors, ids)

    @pytest.mark.parametrize("tag", [[], 123])
    def test_insert_invalid_tag(self, tag, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.insert(gcollection, self.vectors, partition_tag=tag)
