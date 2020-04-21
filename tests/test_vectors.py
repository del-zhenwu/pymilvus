import pytest

from milvus import ParamError, ResponseError
from factorys import records_factory

dim = 128
nq = 100


class TestGetVectorByID:
    def test_get_vector_by_id(self, gcon, gcollection):
        vectors = records_factory(128, 1000)
        ids = [i for i in range(1000)]
        ids_out = gcon.insert(collection_name=gcollection, records=vectors, ids=ids)

        gcon.flush([gcollection])
        vec = gcon.get_vector_by_id(gcollection, ids_out[0])
        # assert vec == vectors[0]

    @pytest.mark.parametrize("v_id", [None, "", [], {"a": 1}, (1, 2)])
    def test_get_vector_by_id_invalid_id(self, v_id, gcon):
        with pytest.raises(ParamError):
            gcon.get_vector_by_id("test_get_vector_by_id_invalid_id", v_id)

    @pytest.mark.parametrize("collection", [None, -1, [], {"a": 1}, (1, 2)])
    def test_get_vector_by_id_invalid_collecton(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.get_vector_by_id("test_get_vector_by_id_invalid_collection", collection)

    def test_get_vector_by_id_non_existent_collection(self, gcon):
        with pytest.raises(ResponseError):
            gcon.get_vector_by_id("non_existent", 1)

    @pytest.mark.parametrize("v_id", [0, 9999])
    def test_get_vector_by_id_non_existent_id(self, v_id, gcon, gcollection):
        with pytest.raises(ResponseError):
            gcon.get_vector_by_id(gcollection, v_id)


class TestDeleteByID:
    def test_delete_by_id_normal(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        ids = gcon.insert(gcollection, vectors)
        gcon.flush([gcollection])

        gcon.delete_by_id(gcollection, ids[0:10])

    @pytest.mark.parametrize("id_", [None, "123", []])
    def test_delete_by_id_invalid_id(self, id_, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.delete_by_id(gcollection, id_)

    @pytest.mark.skip
    def test_delete_by_id_succeed_id(self, gcon, gcollection):
        vectors = records_factory(dim, nq)
        ids = gcon.insert(gcollection, vectors)

        gcon.flush([gcollection])

        ids_exceed = [ids[-1] + 10]
        gcon.delete_by_id(gcollection, ids_exceed)


class TestGetVectorID:
    def test_get_vector_id(self, gcon, gvector):
        info = gcon.collection_info(gvector)

        seg0 = info.partitions_stat[0].segments_stat[0]
        ids = gcon.get_vector_ids(gvector, seg0.segment_name)
        assert isinstance(ids, list)
        assert len(ids) == 10000

    @pytest.mark.parametrize("collection", [None, "", [], {"a": 1}, (1, 2), True, False])
    def test_get_vector_id_invalid_collection(self, collection, gcon):
        with pytest.raises(ParamError):
            gcon.get_vector_ids(collection, "test")

    @pytest.mark.parametrize("segment", [None, "", [], {"a": 1}, (1, 2), True, False])
    def test_get_vector_id_invalid_segment(self, segment, gcon):
        with pytest.raises(ParamError):
            gcon.get_vector_ids("test", segment)

    def test_get_vector_id_non_existent(self, gcon, gvector):
        with pytest.raises(ResponseError):
            gcon.get_vector_ids(gvector, "segment")

        info = gcon.collection_info(gvector)
        seg0 = info.partitions_stat[0].segments_stat[0]
        with pytest.raises(ResponseError):
            gcon.get_vector_ids("test", seg0.segment_name)
