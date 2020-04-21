import pytest

from milvus import ParamError, ResponseError


class TestSegment:
    def test_get_segment_ids(self, gcon, gvector):
        info = gcon.collection_info(gvector)
        seg0 = info.partitions_stat[0].segments_stat[0]
        ids = gcon.get_vector_ids(gvector, seg0.segment_name)
        assert isinstance(ids, list)
        assert len(ids) == 10000

    @pytest.mark.parametrize("collection", [123, None, [], {}, "", True, False])
    @pytest.mark.parametrize("segment", [123, None, [], {}, "", True, False])
    def test_get_segment_invalid_param(self, collection, segment, gcon):
        with pytest.raises(ParamError):
            gcon.get_vector_ids(collection, segment)

    def test_get_segment_non_existent_collection_segment(self, gcon, gcollection):
        with pytest.raises(ResponseError):
            gcon.get_vector_ids("ijojojononsfsfswgsw", "aaa")

        with pytest.raises(ResponseError):
            gcon.get_vector_ids(gcollection, "aaaaaa")
