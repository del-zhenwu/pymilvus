import pytest

from milvus import IndexType
from milvus import ParamError, ResponseError


class TestCreateIndex:
    def test_create_index_normal(self, gcon, gvector):
        try:
            gcon.create_index(gvector, IndexType.IVF_FLAT, {"nlist": 1024})
        except Exception as e:
            pytest.fail("Failed with error: {}".format(e))

    @pytest.mark.parametrize("index,param", [(IndexType.FLAT, {"nlist": 1024}),
                                             (IndexType.IVF_FLAT, {"nlist": 1024}),
                                             (IndexType.IVF_SQ8, {"nlist": 1024}),
                                             (IndexType.IVF_SQ8_H, {"nlist": 1024}),
                                             (IndexType.IVF_PQ, {"m": 16, "nlist": 1024}),
                                             (IndexType.HNSW, {"M": 16, "efConstruction": 500}),
                                             (IndexType.RNSG, {"search_length": 45, "out_degree": 50,
                                                               "candidate_pool_size": 300, "knng": 100}),
                                             (IndexType.ANNOY, {"n_trees": 20})])
    def test_create_index_whole(self, index, param, gcon, gvector):
        mode = gcon._cmd("mode")

        if mode == "GPU" and index in (IndexType.IVF_PQ,):
            pytest.skip("Index {} not support in GPU version".format(index))
        if mode == "CPU" and index in (IndexType.IVF_SQ8_H,):
            pytest.skip("Index {} not support in CPU version".format(index))

        try:
            gcon.create_index(gvector, index, param)
        except Exception as e:
            pytest.fail("Failed with error: {}".format(e))

    def test_create_index_async(self, gcon, gvector):
        future = gcon.create_index(gvector, IndexType.IVFLAT, params={"nlist": 1024}, _async=True)
        future.result()

    def test_create_index_async_callback(self, gcon, gvector):
        def cb(status):
            assert status.OK()

        future = gcon.create_index(gvector, IndexType.IVFLAT, params={"nlist": 1024}, _async=True, _callback=cb)
        future.done()

    @pytest.mark.parametrize("index", [IndexType.INVALID, -1, 100, ""])
    def test_create_index_invalid_index(self, index, gcon, gvector):
        with pytest.raises(ParamError):
            gcon.create_index(gvector, index, {})

    def test_create_index_missing_index(self, gcon, gvector):
        try:
            gcon.create_index(gvector, params={"nlist": 1024})
        except Exception as e:
            pytest.fail("Failed with error: {}".format(e))


class TestDescribeIndex:
    def test_describe_index_normal(self, gcon, gvector):
        try:
            gcon.create_index(gvector, IndexType.IVFLAT, params={"nlist": 1024})
            index = gcon.describe_index(gvector)
            assert index.collection_name == gvector
            assert index.index_type == IndexType.IVFLAT
            assert index.params == {"nlist": 1024}
        except Exception as e:
            pytest.fail("Failed with error: {}".format(e))

    @pytest.mark.parametrize("collection", [123, None, []])
    def test_describe_index_invalid_name(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.describe_index(collection)

    def test_describe_index_non_existent(self, gcon):
        with pytest.raises(ResponseError):
            gcon.describe_index("non_existent")


class TestDropIndex:
    def test_drop_index_normal(self, gcon, gvector):
        try:
            gcon.create_index(gvector, IndexType.IVFLAT, {"nlist": 1024})
            gcon.drop_index(gvector)
        except Exception as e:
            pytest.fail("Failed with error: {}".format(e))

    @pytest.mark.parametrize("collection", [123, None, []])
    def test_drop_index_invalid_name(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.drop_index(collection)

    def test_drop_index_non_existent(self, gcon):
        with pytest.raises(ResponseError):
            gcon.drop_index("non_existent")
