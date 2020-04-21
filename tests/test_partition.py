import pytest

from milvus import ParamError, ResponseError


class TestCreatePartition:
    def test_create_partition_normal(self, gcon, gcollection):
        try:
            gcon.create_partition(gcollection, "new_tag")
        except Exception:
            pytest.fail("Error raise")

    @pytest.mark.parametrize("tag", [[], 1234353, {}, [1, 2]])
    def test_create_partition_invalid_tag(self, tag, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.create_partition(gcollection, tag)

    @pytest.mark.skip(reason="Bug here. See #1762(https://github.com/milvus-io/milvus/issues/1762)")
    def test_create_partition_default(self, gcon, gcollection):
        with pytest.raises(ResponseError):
            gcon.create_partition(gcollection, "_default")

    def test_create_partition_repeat(self, gcon, gcollection):
        try:
            gcon.create_partition(gcollection, "tag01")
        except Exception:
            pytest.fail("Error raise")

        with pytest.raises(ResponseError):
            gcon.create_partition(gcollection, "tag01")


class TestShowPartitions:
    def test_show_partitions_normal(self, gcon, gcollection):
        try:
            gcon.create_partition(gcollection, "tag01")
            partitions = gcon.show_partitions(gcollection)
            for partition in partitions:
                assert partition.collection_name == gcollection
                assert partition.tag in ("tag01", "_default")
        except:
            pytest.fail("Error raise")


class TestDropPartition:
    def test_drop_partition(self, gcon, gcollection):
        try:
            gcon.create_partition(gcollection, "tag01")
            gcon.drop_partition(gcollection, "tag01")
        except:
            pytest.fail("Error raise")

    @pytest.mark.skip
    @pytest.mark.parametrize("tag", [[], None, 123, {}])
    def test_drop_partition_invalid_tag(self, tag, gcon, gcollection):
        with pytest.raises(ParamError):
            gcon.drop_partition(gcollection, tag)

    def test_drop_partition_non_existent(self, gcon, gcollection):
        with pytest.raises(ResponseError):
            gcon.drop_partition(gcollection, "non_existent")
