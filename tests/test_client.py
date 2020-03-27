import logging
import numpy as np
import time
import random
import pytest
import sys
import ujson

sys.path.append('.')

from faker import Faker

from milvus import IndexType, MetricType, Prepare, Milvus, Status, ParamError, NotConnectError
from milvus.client.abstract import CollectionSchema, TopKQueryResult
from milvus.client.check import check_pass_param
from milvus.client.hooks import BaseSearchHook

from factorys import (
    collection_schema_factory,
    records_factory,
    fake
)
from milvus.grpc_gen import milvus_pb2

logging.getLogger('faker').setLevel(logging.ERROR)
LOGGER = logging.getLogger(__name__)
faker = Faker(locale='en_US')

dim = 128
nb = 2000
nq = 10


class TestVector:
    @pytest.mark.skip
    def test_insert_with_numpy(self, gcon, gcollection):
        vectors = np.random.rand(nq, dim).astype(np.float32)
        param = {
            'collection_name': gcollection,
            'records': vectors
        }

        res, ids = gcon.insert(**param)
        assert res.OK()
        assert isinstance(ids, list)
        assert len(ids) == nq


class TestPrepare:
    def test_insert_numpy_array(self, gcon ,gcollection):
        vectors = np.random.rand(10000, 128)
        status, ids = gcon.insert(gcollection, vectors)
        assert status.OK(), status.message

@pytest.mark.skip(reason="crud branch")
class TestSearchByID:
    def test_search_by_id_normal(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors)

        assert status.OK()

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        status, result = gcon.search_by_id(gcollection, 2, 10, ids[0])
        assert status.OK()

        print(result)

        assert 1 == len(result)
        assert 2 == len(result[0])
        assert ids[0] == result[0][0].id

    def test_search_by_id_with_partitions(self, gcon, gcollection):
        tag = "search_by_id_partitions_tag"

        status = gcon.create_partition(gcollection, tag)
        assert status.OK()

        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors, partition_tag=tag)
        assert status.OK()

        time.sleep(2)

        status, result = gcon.search_by_id(gcollection, 2, 10, ids[0], partition_tag_array=[tag])
        assert status.OK()

        assert 1 == len(result)
        assert 2 == len(result[0])
        assert ids[0] == result[0][0].id

    def test_search_by_id_with_wrong_param(self, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.search_by_id(gcollection, 'x', 1, 1)

        with pytest.raises(ParamError):
            gcon.search_by_id(gcollection, 1, '1', 1)

        with pytest.raises(ParamError):
            gcon.search_by_id(gcollection, 1, 1, 'aaa')

        status, _ = gcon.search_by_id(gcollection, -1, 1, 1)
        assert not status.OK()

        status, _ = gcon.search_by_id(gcollection, 1, -1, 1)
        assert not status.OK()

    @pytest.mark.skip(reason="except empty result, return result with -1 id instead")
    def test_search_by_id_with_exceed_id(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors)
        assert status.OK()

        status, result = gcon.search_by_id(gcollection, 2, 10, ids[0] + 100)
        assert status.OK()
        print(result)
        assert 0 == len(result)


class TestCmd:
    versions = ("0.5.3", "0.6.0", "0.7.0")

    def test_client_version(self, gcon):
        try:
            import milvus
            assert gcon.client_version() == milvus.__version__
        except ImportError:
            assert False, "Import error"

    def test_server_version(self, gcon):
        _, version = gcon.server_version()
        assert version in self.versions

    def test_server_status(self, gcon):
        _, status = gcon.server_status()
        assert status in ("OK", "ok")

    def test_cmd(self, gcon):
        _, info = gcon._cmd("version")
        assert info in self.versions

        _, info = gcon._cmd("status")
        assert info in ("OK", "ok")


class TestChecking:

    @pytest.mark.parametrize(
        "key_, value_",
        [("ids", [1, 2]), ("nprobe", 12), ("nlist", 4096), ("cmd", 'OK')]
    )
    def test_param_check_normal(self, key_, value_):
        try:
            check_pass_param(**{key_: value_})
        except Exception:
            assert False

    @pytest.mark.parametrize(
        "key_, value_",
        [("ids", []), ("nprobe", "aaa"), ("nlist", "aaa"), ("cmd", 123)]
    )
    def test_param_check_error(self, key_, value_):
        with pytest.raises(ParamError):
            check_pass_param(**{key_: value_})


class TestQueryResult:
    query_vectors = [[random.random() for _ in range(128)] for _ in range(200)]

    def _get_response(self, gcon, gvector, topk, nprobe, nq):
        search_param = {
            "nprobe": nprobe
        }

        return gcon.search(gvector, topk, self.query_vectors[:nq], params=search_param)

    def test_search_result(self, gcon, gvector):
        try:
            status, results = self._get_response(gcon, gvector, 2, 1, 1)
            assert status.OK()

            # test get_item
            shape = results.shape

            # test TopKQueryResult slice
            rows = results[:1]

            # test RowQueryResult
            row = results[shape[0] - 1]

            # test RowQueryResult slice
            items = row[:1]

            # test iter
            for topk_result in results:
                for item in topk_result:
                    print(item)

            # test len
            len(results)
            # test print
            print(results)

            # test result for nq = 10, topk = 10
            status, results = self._get_response(gcon, gvector, 10, 10, 10)
            print(results)
        except Exception:
            assert False

    def test_search_in_files_result(self, gcon, gvector):
        try:
            search_param = {
                "nprobe": 1
            }

            for index in range(1000):
                status, results = \
                    gcon.search_in_files(collection_name=gvector, top_k=1,
                                         file_ids=[str(index)], query_records=self.query_vectors, params=search_param)
                if status.OK():
                    break

            # test get_item
            shape = results.shape
            item = results[shape[0] - 1][shape[1] - 1]

            # test iter
            for topk_result in results:
                for item in topk_result:
                    print(item)

            # test len
            len(results)
            # test print
            print(results)
        except Exception:
            assert False

    def test_empty_result(self, gcon, gcollection):
        status, results = self._get_response(gcon, gcollection, 3, 3, 3)
        shape = results.shape

        for topk_result in results:
            for item in topk_result:
                print(item)


class TestDeleteByID:
    def test_delete_by_id_normal(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors)
        assert status.OK()

        time.sleep(2)

        status = gcon.delete_by_id(gcollection, ids[0:10])
        assert status.OK()

    def test_delete_by_id_wrong_param(self, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.delete_by_id(gcollection, "aaa")

    @pytest.mark.skip
    def test_delete_by_id_succeed_id(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        status, ids = gcon.insert(gcollection, vectors)
        assert status.OK()

        time.sleep(2)

        ids_exceed = [ids[-1] + 10]
        status = gcon.delete_by_id(gcollection, ids_exceed)
        assert not status.OK()


class TestFlush:
    def test_flush(self, gcon):
        collection_param = {
            "collection_name": '',
            "dimension": dim
        }

        collection_list = ["test_flush_1", "test_flush_2", "test_flush_3"]
        vectors = records_factory(dim, nq)
        for collection in collection_list:
            collection_param["collection_name"] = collection

            gcon.create_collection(collection_param)

            gcon.insert(collection, vectors)

        status = gcon.flush(collection_list)
        assert status.OK()

        for collection in collection_list:
            gcon.drop_collection(collection)

    def test_flush_with_none(self, gcon, gcollection):
        collection_param = {
            "collection_name": '',
            "dimension": dim
        }

        collection_list = ["test_flush_1", "test_flush_2", "test_flush_3"]
        vectors = records_factory(dim, nq)
        for collection in collection_list:
            collection_param["collection_name"] = collection

            gcon.create_collection(collection_param)

            gcon.insert(collection, vectors)

        status = gcon.flush()
        assert status.OK(), status.message

        for collection in collection_list:
            gcon.drop_collection(collection)


class TestCompact:
    def test_compact_normal(self, gcon, gcollection):
        vectors = [[random.random() for _ in range(128)] for _ in range(10000)]
        status, ids = gcon.add_vectors(collection_name=gcollection, records=vectors)
        assert status.OK()

        status = gcon.compact(gcollection)
        assert status.OK(), status.message

    def test_compact_after_delete(self, gcon, gcollection):
        vectors = [[random.random() for _ in range(128)] for _ in range(10000)]
        status, ids = gcon.insert(collection_name=gcollection, records=vectors)
        assert status.OK(), status.message

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        status = gcon.delete_by_id(gcollection, ids[100:1000])
        assert status, status.message

        status = gcon.compact(gcollection)
        assert status.OK(), status.message

    def test_compact_with_empty_collection(self, gcon, gcollection):
        status = gcon.compact(gcollection)
        assert status.OK(), status.message

    def test_compact_with_non_exist_name(self, gcon):
        status = gcon.compact(collection_name="die333")
        assert not status.OK()

    def test_compact_with_invalid_name(self, gcon):
        with pytest.raises(ParamError):
            gcon.compact(collection_name=124)


class TestCollectionInfo:
    def test_collection_info_normal(self, gcon, gcollection):
        for _ in range(10):
            records = records_factory(128, 10000)
            status, _ = gcon.insert(gcollection, records)
            assert status.OK()

        gcon.flush([gcollection])

        status, _ = gcon.collection_info(gcollection, timeout=None)
        assert status.OK()

    def test_collection_info_with_partitions(self, gcon, gcollection):
        for i in range(5):
            partition_tag = "tag_{}".format(i)

            status = gcon.create_partition(gcollection, partition_tag)
            assert status.OK(), status.message

            for j in range(3):
                records = records_factory(128, 10000)
                status, _ = gcon.insert(gcollection, records, partition_tag=partition_tag)
                assert status.OK(), status.message

        status = gcon.flush([gcollection])
        assert status.OK(), status.message

        status, _ = gcon.collection_info(gcollection, timeout=None)
        assert status.OK(), status.message

    def test_collection_info_with_empty_collection(self, gcon, gcollection):
        status, _ = gcon.collection_info(gcollection)

        assert status.OK(), status.message

    def test_collection_info_with_non_exist_collection(self, gcon):
        status, _ = gcon.collection_info("Xiaxiede")
        assert not status.OK(), status.message
