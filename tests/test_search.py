import pytest
import random

from milvus import ParamError, ResponseError
from factorys import records_factory

dim = 128
nq = 100


class TestSearch:
    topk = random.randint(1, 10)
    query_records = records_factory(dim, nq)
    search_param = {
        "nprobe": 10
    }

    def test_search_normal(self, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': self.topk,
            'params': self.search_param
        }
        try:
            results = gcon.search(**param)
            assert len(results) == nq
            assert len(results[0]) == self.topk
            assert results.shape[0] == nq
            assert results.shape[1] == self.topk
        except:
            pytest.fail("Exception raise")

    def test_search_default_partition(self, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': self.topk,
            'partition_tags': ["_default"],
            'params': self.search_param
        }
        results = gcon.search(**param)
        assert len(results) == nq
        assert len(results[0]) == self.topk
        assert results.shape[0] == nq
        assert results.shape[1] == self.topk

    def test_search_async_normal(self, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': self.topk,
            'params': self.search_param,
            '_async': True
        }
        future = gcon.search(**param)
        results = future.result()

        assert len(results) == nq
        assert len(results[0]) == self.topk

        assert results.shape[0] == nq
        assert results.shape[1] == self.topk

    def test_search_async_callback(self, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': self.topk,
            'params': self.search_param,
            '_async': True
        }

        def cb(results):
            assert len(results) == nq
            assert len(results[0]) == self.topk
            assert results.shape[0] == nq
            assert results.shape[1] == self.topk

        future = gcon.search(_callback=cb, **param)
        future.done()

    @pytest.mark.parametrize("query_records", [[], None, "", 123])
    def test_search_invalid_query_records(self, query_records, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': query_records,
            'top_k': self.topk,
            'params': self.search_param,
        }
        with pytest.raises(ParamError):
            gcon.search(**param)

    @pytest.mark.parametrize("params", [[], "", 123])
    def test_search_invalid_params(self, params, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': self.topk,
            'params': params,
        }
        with pytest.raises(ParamError):
            gcon.search(**param)

    @pytest.mark.parametrize("topk", [[], "", None, {}])
    def test_search_invalid_topk(self, topk, gcon, gvector):
        param = {
            'collection_name': gvector,
            'query_records': self.query_records,
            'top_k': topk,
            'params': self.search_param,
        }
        with pytest.raises(ParamError):
            gcon.search(**param)


class TestSearchInFiles:
    def test_search_in_files_normal(self, gcon, gvector):
        search_param = {
            "nprobe": 10
        }

        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        for i in range(5000):
            try:
                gcon.search_in_files(gvector, file_ids=[i], top_k=1,
                                             query_records=query_vectors, params=search_param)
                return
            except ResponseError:
                continue
            except Exception:
                pytest.fail("Unknown error raise")

        assert False

    def test_search_in_files_async(self, gcon, gvector):
        search_param = {
            "nprobe": 10
        }

        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        for i in range(5000):
            future = gcon.search_in_files(gvector, file_ids=[i], top_k=1, query_records=query_vectors,
                                          params=search_param, _async=True)
            try:
                future.result()
                return
            except ResponseError:
                continue
            except Exception:
                pytest.fail("Unknown error raise")

        assert False

    def test_search_in_files_async_callback(self, gcon, gvector):
        def cb(status, results):
            print("Search status: ", status)

        search_param = {
            "nprobe": 10
        }

        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        for i in range(5000):
            future = gcon.search_in_files(gvector, file_ids=[i], top_k=1, query_records=query_vectors,
                                          params=search_param, _async=True, _callback=cb)
            try:
                future.result()
                return
            except ResponseError:
                continue
            except Exception:
                pytest.fail("Unknown error raise")

        assert False

    @pytest.mark.parametrize("collection", [[], None, "", 123])
    def test_search_in_files_invalid_collection(self, collection, gcon):
        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        with pytest.raises(ParamError):
            gcon.search_in_files(collection, file_ids=[1], top_k=1, query_records=query_vectors, params={"nprobe": 1})

    @pytest.mark.parametrize("ids", [[], None, "", 123])
    def test_search_in_files_invalid_file_ids(self, ids, gcon):
        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        with pytest.raises(ParamError):
            gcon.search_in_files("test", file_ids=ids, top_k=1, query_records=query_vectors, params={"nprobe": 1})

    @pytest.mark.parametrize("topk", [[], None, "", {}, True, False])
    def test_search_in_files_invalid_topk(self, topk, gcon):
        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        with pytest.raises(ParamError):
            gcon.search_in_files("test", file_ids=[1], top_k=topk, query_records=query_vectors, params={"nprobe": 1})

    @pytest.mark.parametrize("records", [[], None, "", 123, True, False])
    def test_search_in_files_invalid_records(self, records, gcon):
        with pytest.raises(ParamError):
            gcon.search_in_files("test", file_ids=[1], top_k=1, query_records=records, params={"nprobe": 1})

    @pytest.mark.parametrize("param", [[], "", 123, (), set(), True, False])
    def test_search_in_files_invalid_param(self, param, gcon):
        query_vectors = [[random.random() for _ in range(128)] for _ in range(100)]
        with pytest.raises(ParamError):
            gcon.search_in_files("test", file_ids=[1], top_k=1, query_records=query_vectors, params=param)
