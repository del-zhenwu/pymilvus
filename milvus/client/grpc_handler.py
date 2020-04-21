import datetime
import functools
from urllib.parse import urlparse
import logging
import threading

import grpc
from grpc._cython import cygrpc

from ..grpc_gen import milvus_pb2_grpc
from ..grpc_gen import milvus_pb2 as grpc_types
from .abstract import ConnectIntf, CollectionSchema, IndexParam, PartitionParam, TopKQueryResult, CollectionInfo
from .prepare import Prepare
from .types import MetricType, Status
from .check import (
    int_or_str,
    is_legal_host,
    is_legal_port,
)

from .asynch import Future, SearchFuture, InsertFuture, CreateIndexFuture, CompactFuture, FlushFuture

from .hooks import BaseSearchHook
from .client_hooks import SearchHook
from .exceptions import ParamError, NotConnectError, ResponseError
from ..settings import DefaultConfig as config
from . import __version__

LOGGER = logging.getLogger(__name__)


def error_handler(func):
    @functools.wraps(func)
    def handler(*args, **kwargs):
        try:
            results = func(*args, **kwargs)
            if isinstance(results, Future):
                return results
            status = results[0] if isinstance(results, tuple) else results
            if status.code != 0:
                raise ResponseError(status.code, status.message)
            if isinstance(results, tuple):
                return results[1]
        except Exception as e:
            LOGGER.error("{}\n{}".format(func.__name__, e))
            raise
        # except grpc.FutureTimeoutError as e:
        #     LOGGER.error("{}\n{}".format(func.__name__, e))
        #     status = Status(Status.UNEXPECTED_ERROR, message='Request timeout')
        # except grpc.RpcError as e:
        #     LOGGER.error("{}\n{}".format(func.__name__, e))
        #     status = Status(e.code(), message='Error occurred. {}'.format(e.details()))

    return handler


def set_uri(host, port, uri):
    if host is not None:
        _port = port if port is not None else config.GRPC_PORT
        _host = host
    elif port is None:
        try:
            # Ignore uri check here
            # if not is_legal_uri(_uri):
            #     raise ParamError("uri {} is illegal".format(_uri))
            #
            # If uri is empty (None or '') use default uri instead
            # (the behavior may change in the future)
            # _uri = urlparse(_uri) if _uri is not None else urlparse(config.GRPC_URI)
            _uri = urlparse(uri) if uri else urlparse(config.GRPC_URI)
            _host = _uri.hostname
            _port = _uri.port
        except (AttributeError, ValueError, TypeError) as e:
            raise ParamError("uri is illegal: {}".format(e))
    else:
        raise ParamError("Param is not complete. Please invoke as follow:\n"
                         "\t(host = ${HOST}, port = ${PORT})\n"
                         "\t(uri = ${URI})\n")

    if not is_legal_host(_host) or not is_legal_port(_port):
        raise ParamError("host or port is illeagl")

    return "{}:{}".format(str(_host), str(_port))


def connect(addr, timeout):
    channel = grpc.insecure_channel(
        addr,
        options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                 (cygrpc.ChannelArgKey.max_receive_message_length, -1)]
    )
    try:
        ft = grpc.channel_ready_future(channel)
        ft.result(timeout=timeout)
        return True
    except grpc.FutureTimeoutError:
        raise NotConnectError('Fail connecting to server on {}. Timeout'.format(addr))
    except grpc.RpcError as e:
        raise NotConnectError("Connect error: <{}>".format(e))
    # Unexpected error
    except Exception as e:
        raise NotConnectError("Error occurred when trying to connect server:\n"
                              "\t<{}>".format(str(e)))
    finally:
        ft.cancel()
        ft.__del__()
        channel.__del__()


class GrpcHandler(ConnectIntf):
    def __init__(self, host=None, port=None, **kwargs):
        self._stub = None
        self._uri = None
        self.status = None
        self._connected = False

        # client hook
        self._search_hook = SearchHook()
        self._search_file_hook = SearchHook()

        # set server uri if object is initialized with parameter
        _uri = kwargs.get("uri", None)
        if host or port or _uri:
            self._uri = set_uri(host, port, uri=_uri)
            self._setup(kwargs.get("pre_ping", False))

    def __str__(self):
        attr_list = ['%s=%r' % (key, value)
                     for key, value in self.__dict__.items() if not key.startswith('_')]
        return '<Milvus: {}>'.format(', '.join(attr_list))

    def __enter__(self):
        self._setup()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self._stub

    def _setup(self, pre_ping=False):
        """
        Create a grpc channel and a stub

        :raises: NotConnectError

        """
        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._set_channel())
        self.status = Status()

    def _set_channel(self):
        """
        Set grpc channel. Use default server uri if uri is not set.
        """

        # set transport unlimited
        return grpc.insecure_channel(
            self._uri,
            # self._uri or config.GRPC_ADDRESS,
            options=[(cygrpc.ChannelArgKey.max_send_message_length, -1),
                     (cygrpc.ChannelArgKey.max_receive_message_length, -1)]
        )

    def set_hook(self, **kwargs):
        """
        specify client hooks.
        The client hooks are used in methods which interact with server.
        Use key-value method to set hooks. Supported hook setting currently is as follow.

            search hook,
            search-in-file hook

        """

        # config search hook
        _search_hook = kwargs.get('search', None)
        if _search_hook:
            if not isinstance(_search_hook, BaseSearchHook):
                raise ParamError("search hook must be a subclass of `BaseSearchHook`")

            self._search_hook = _search_hook

        _search_file_hook = kwargs.get('search_in_file', None)
        if _search_file_hook:
            if not isinstance(_search_file_hook, BaseSearchHook):
                raise ParamError("search hook must be a subclass of `BaseSearchHook`")

            self._search_file_hook = _search_file_hook

    @property
    def server_address(self):
        """
        Server network address
        """
        return self._uri

    def connect(self, host=None, port=None, uri=None, timeout=1):
        """
        Connect method should be called before any operations.
        Server will be connected after connect return OK

        :type  host: str
        :type  port: str
        :type  uri: str
        :type  timeout: float
        :param host: (Optional) host of the server, default host is 127.0.0.1
        :param port: (Optional) port of the server, default port is 19530
        :param uri: (Optional) only support tcp proto now, default uri is

                `tcp://127.0.0.1:19530`

        :param timeout: (Optional) connection timeout, default timeout is 3000ms

        :return: Status, indicate if connect is successful
        :rtype: Status
        
        :raises: NotConnectError
        """
        if self.connected() and self._connected:
            return Status(message="You have already connected {} !".format(self._uri),
                          code=Status.CONNECT_FAILED)

        # TODO: Here may cause bug: IF user has already connected a server but server is down,
        # client may connect to a new server. It's a undesirable behavior.

        if (host or port or uri) or not self._uri:
            # if self._uri and self._uri != self._set_uri(host, port, uri=uri):
            #     return Status(message="The server address is set as {}, "
            #                           "you cannot connect other server".format(self._uri),
            #                   code=Status.CONNECT_FAILED)
            self._uri = set_uri(host, port, uri=uri)
            connect(self._uri, timeout)

        self._stub = milvus_pb2_grpc.MilvusServiceStub(self._set_channel())
        self.status = Status()
        return self.status

    def connected(self):
        """
        Check if client is connected to the server

        :return: if client is connected
        :rtype: bool
        """
        return True if self.status and self.status.OK() else False

    def disconnect(self):
        """
        Disconnect with the server and distroy the channel

        :return: Status, indicate if disconnect is successful
        :rtype: Status
        """
        # After closeing, a exception stack trace is printed from a background thread and
        # no exception is thrown in the main thread, issue is under test and not done yet
        # checkout https://github.com/grpc/grpc/issues/18995
        # Also checkout Properly Specify Channel.close Behavior in Python:
        # https://github.com/grpc/grpc/issues/19235
        if not self.connected():
            raise NotConnectError('Please connect to the server first!')

        # try:
        #     self._channel.close()
        # except Exception as e:
        #     LOGGER.error(e)
        #     return Status(code=Status.CONNECT_FAILED, message='Disconnection failed')

        self.status = None
        self._stub = None
        return Status(message='Disconnect successfully')

    def server_version(self, timeout=10):
        """
        Provide server version

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        return self._cmd(cmd='version', timeout=timeout)

    def server_status(self, timeout=10):
        """
        Provide server status

        :return:
            Status: indicate if operation is successful

            str : Server version

        :rtype: (Status, str)
        """
        return self._cmd(cmd='status', timeout=timeout)

    @error_handler
    def _cmd(self, cmd, timeout=10):
        cmd = Prepare.cmd(cmd)
        rf = self._stub.Cmd.future(cmd)
        response = rf.result(timeout=timeout)
        rf.__del__()
        if response.status.error_code == 0:
            return Status(message='Success!'), response.string_reply

        return Status(code=response.status.error_code, message=response.status.reason), None

    @error_handler
    def create_collection(self, collection_name, dimension, index_file_size, metric_type, param, timeout=10):
        """
        Create collection

        :type  param: dict or TableSchema
        :param param: Provide collection information to be created

                `example param={'collection_name': 'name',
                                'dimension': 16,
                                'index_file_size': 1024 (default)，
                                'metric_type': Metric_type.L2 (default)
                                }`

                `OR using Prepare.collection_schema to create param`

        :param timeout: timeout, The unit is seconds
        :type  timeout: double

        :return: Status, indicate if operation is successful
        :rtype: Status
        """

        collection_schema = Prepare.collection_schema(collection_name, dimension, index_file_size, metric_type, param)

        rf = self._stub.CreateCollection.future(collection_schema)
        status = rf.result(timeout=timeout)
        rf.__del__()
        if status.error_code == 0:
            return Status(message='Create collection successfully!')

        LOGGER.error(status)
        return Status(code=status.error_code, message=status.reason)

    @error_handler
    def has_collection(self, collection_name, timeout=10):
        """

        This method is used to test collection existence.

        :param collection_name: collection name is going to be tested.
        :type  collection_name: str
        :param timeout: time waiting for server response
        :type  timeout: int

        :return:
            Status: indicate if vectors inserted successfully
            bool if given collection_name exists

        """

        collection_name = Prepare.collection_name(collection_name)

        rf = self._stub.HasCollection.future(collection_name)
        reply = rf.result(timeout=timeout)
        rf.__del__()
        if reply.status.error_code == 0:
            return Status(), reply.bool_reply

        return Status(code=reply.status.error_code, message=reply.status.reason), False

    @error_handler
    def describe_collection(self, collection_name, timeout=10):
        """
        Show collection information

        :type  collection_name: str
        :param collection_name: which collection to be shown

        :returns: (Status, collection_schema)
            Status: indicate if query is successful
            collection_schema: return when operation is successful
        :rtype: (Status, TableSchema)
        """
        collection_name = Prepare.collection_name(collection_name)
        rf = self._stub.DescribeCollection.future(collection_name)
        response = rf.result(timeout=timeout)
        rf.__del__()

        if response.status.error_code == 0:
            collection = CollectionSchema(
                collection_name=response.collection_name,
                dimension=response.dimension,
                index_file_size=response.index_file_size,
                metric_type=MetricType(response.metric_type)
            )
            return Status(message='Describe collection successfully!'), collection

        LOGGER.error(response.status)
        return Status(code=response.status.error_code, message=response.status.reason), None

    @error_handler
    def count_collection(self, collection_name, timeout=30):
        """
        obtain vector number in collection

        :type  collection_name: str
        :param collection_name: target collection name.

        :returns:
            Status: indicate if operation is successful

            res: int, collection row count
        """

        collection_name = Prepare.collection_name(collection_name)

        rf = self._stub.CountCollection.future(collection_name)
        response = rf.result(timeout=timeout)
        rf.__del__()
        if response.status.error_code == 0:
            return Status(message='Success!'), response.collection_row_count

        return Status(code=response.status.error_code, message=response.status.reason), None

    @error_handler
    def show_collections(self, timeout=10):

        cmd = Prepare.cmd('show_collections')
        rf = self._stub.ShowCollections.future(cmd)
        response = rf.result(timeout=timeout)
        rf.__del__()
        if response.status.error_code == 0:
            return Status(message='Show collections successfully!'), \
                   [name for name in response.collection_names if len(name) > 0]
        return Status(response.status.error_code, message=response.status.reason), []

    @error_handler
    def show_collection_info(self, collection_name, timeout=10):
        request = grpc_types.CollectionName(collection_name=collection_name)

        rf = self._stub.ShowCollectionInfo.future(request)
        response = rf.result(timeout=timeout)
        rf.__del__()
        rpc_status = response.status

        if rpc_status.error_code == 0:
            return Status(), CollectionInfo(response)

        return Status(rpc_status.error_code, rpc_status.reason), None

    @error_handler
    def preload_collection(self, collection_name, timeout=None):
        collection_name = Prepare.collection_name(collection_name)
        status = self._stub.PreloadCollection.future(collection_name).result(timeout=timeout)
        return Status(code=status.error_code, message=status.reason)

    @error_handler
    def drop_collection(self, collection_name, timeout=20):
        collection_name = Prepare.collection_name(collection_name)

        rf = self._stub.DropCollection.future(collection_name)
        status = rf.result(timeout=timeout)
        rf.__del__()
        if status.error_code == 0:
            return Status(message='Delete collection successfully!')
        return Status(code=status.error_code, message=status.reason)

    @error_handler
    def insert(self, collection_name, records, ids=None, partition_tag=None, params=None, timeout=None, **kwargs):
        insert_param = kwargs.get('insert_param', None)

        if insert_param and not isinstance(insert_param, grpc_types.InsertParam):
            raise ParamError("The value of key 'insert_param' is invalid")

        body = insert_param if insert_param \
            else Prepare.insert_param(collection_name, records, partition_tag, ids, params)

        rf = self._stub.Insert.future(body)
        if kwargs.get("_async", False) is True:
            cb = kwargs.get("callback", None)
            return InsertFuture(rf, cb)

        response = rf.result(timeout=timeout)
        rf.__del__()
        if response.status.error_code == 0:
            return Status(message='Add vectors successfully!'), list(response.vector_id_array)

        return Status(code=response.status.error_code, message=response.status.reason), []

    @error_handler
    def get_vector_by_id(self, collection_name, v_id, timeout=10):
        request = grpc_types.VectorIdentity(collection_name=collection_name, id=v_id)

        rf = self._stub.GetVectorByID.future(request)
        response = rf.result(timeout=timeout)
        rf.__del__()
        status = response.status
        if status.error_code == 0:
            status = Status(message="Obtain vector successfully")
            return status, \
                   bytes(response.vector_data.binary_data) or list(response.vector_data.float_data)

        return Status(code=status.error_code, message=status.reason), []

    @error_handler
    def get_vector_ids(self, collection_name, segment_name, timeout=10):
        request = grpc_types.GetVectorIDsParam(collection_name=collection_name, segment_name=segment_name)

        rf = self._stub.GetVectorIDs.future(request)
        response = rf.result(timeout=timeout)
        rf.__del__()

        if response.status.error_code == 0:
            return Status(), list(response.vector_id_array)
        return Status(response.status.error_code, response.status.reason), []

    @error_handler
    def create_index(self, collection_name, index_type=None, params=None, timeout=None, **kwargs):
        index_param = Prepare.index_param(collection_name, index_type, params)
        # status = self._stub.CreateIndex.future(index_param).result(timeout=timeout)
        future = self._stub.CreateIndex.future(index_param, timeout=timeout)
        if kwargs.get('_async', False):
            cb = kwargs.get("callback", None)
            return CreateIndexFuture(future, cb)
        status = future.result(timeout=timeout)
        future.__del__()

        if status.error_code == 0:
            return Status(message='Build index successfully!')
        return Status(code=status.error_code, message=status.reason)

    @error_handler
    def describe_index(self, collection_name, timeout=10):

        collection_name = Prepare.collection_name(collection_name)

        rf = self._stub.DescribeIndex.future(collection_name)
        index_param = rf.result(timeout=timeout)
        rf.__del__()

        status = index_param.status

        if status.error_code == 0:
            return Status(message="Successfully"), \
                   IndexParam(index_param.collection_name,
                              index_param.index_type,
                              index_param.extra_params)

        return Status(code=status.error_code, message=status.reason), None

    @error_handler
    def drop_index(self, collection_name, timeout=10):
        collection_name = Prepare.collection_name(collection_name)
        rf = self._stub.DropIndex.future(collection_name)
        status = rf.result(timeout=timeout)
        rf.__del__()
        # status = self._stub.DropIndex.future(collection_name).result(timeout=timeout)
        return Status(code=status.error_code, message=status.reason)

    @error_handler
    def create_partition(self, collection_name, partition_tag, timeout=10):
        request = Prepare.partition_param(collection_name, partition_tag)
        rf = self._stub.CreatePartition.future(request)
        response = rf.result(timeout=timeout)
        rf.__del__()
        return Status(code=response.error_code, message=response.reason)

    @error_handler
    def show_partitions(self, collection_name, timeout=10):
        request = Prepare.collection_name(collection_name)

        rf = self._stub.ShowPartitions.future(request)
        response = rf.result(timeout=timeout)
        rf.__del__()
        # response = self._stub.ShowPartitions.future(request).result(timeout=timeout)
        status = response.status
        if status.error_code == 0:
            partition_list = [PartitionParam(collection_name, p) for p in response.partition_tag_array]
            return Status(), partition_list

        return Status(code=status.error_code, message=status.reason), []

    @error_handler
    def drop_partition(self, collection_name, partition_tag, timeout=10):
        request = grpc_types.PartitionParam(collection_name=collection_name, tag=partition_tag)

        rf = self._stub.DropPartition.future(request)
        response = rf.result(timeout=timeout)
        rf.__del__()
        # response = self._stub.DropPartition.future(request).result(timeout=timeout)
        return Status(code=response.error_code, message=response.reason)

    @error_handler
    def search(self, collection_name, top_k, query_records, partition_tags=None, params=None, **kwargs):
        request = Prepare.search_param(collection_name, top_k, query_records, partition_tags, params)

        self._search_hook.pre_search()
        if kwargs.get("_async", False) is True:
            timeout = kwargs.get("timeout", None)
            future = self._stub.Search.future(request, timeout=timeout)

            func = kwargs.get("_callback", None)
            return SearchFuture(future, func)
        ft = self._stub.Search.future(request)
        response = ft.result()
        ft.__del__()
        self._search_hook.aft_search()

        if self._search_hook.on_response():
            return response

        if response.status.error_code != 0:
            return Status(code=response.status.error_code,
                          message=response.status.reason), []

        resutls = self._search_hook.handle_response(response)
        return Status(message='Search vectors successfully!'), resutls

    @error_handler
    def search_in_files(self, collection_name, file_ids, query_records, top_k, params, **kwargs):
        file_ids = list(map(int_or_str, file_ids))
        infos = Prepare.search_vector_in_files_param(
            collection_name, query_records, top_k, file_ids, params
        )

        self._search_file_hook.pre_search()

        if kwargs.get("_async", False) is True:
            timeout = kwargs.get("timeout", None)
            future = self._stub.SearchInFiles.future(infos, timeout=timeout)

            func = kwargs.get("_callback", None)
            return SearchFuture(future, func)

        response = self._stub.SearchInFiles(infos)
        self._search_file_hook.aft_search()

        if self._search_file_hook.on_response():
            return response

        if response.status.error_code != 0:
            return Status(code=response.status.error_code,
                          message=response.status.reason), []

        return Status(message='Search vectors successfully!'), \
               self._search_file_hook.handle_response(response)

    @error_handler
    def delete_by_id(self, collection_name, id_array, timeout=None):
        request = Prepare.delete_by_id_param(collection_name, id_array)

        rf = self._stub.DeleteByID.future(request)
        status = rf.result(timeout=timeout)
        rf.__del__()
        # status = self._stub.DeleteByID.future(request).result(timeout=timeout)
        return Status(code=status.error_code, message=status.reason)

    @error_handler
    def flush(self, collection_name_array, timeout=None, **kwargs):
        request = Prepare.flush_param(collection_name_array)
        future = self._stub.Flush.future(request, timeout=timeout)
        if kwargs.get("_async", False):
            cb = kwargs.get("_callback", None)
            return FlushFuture(future, cb)
        response = future.result(timeout=timeout)
        future.__del__()
        return Status(code=response.error_code, message=response.reason)

    @error_handler
    def compact(self, collection_name, timeout, **kwargs):
        request = Prepare.compact_param(collection_name)
        future = self._stub.Compact.future(request, timeout=timeout)
        if kwargs.get("_async", False):
            cb = kwargs.get("_callback", None)
            return CompactFuture(future, cb)
        response = future.result(timeout=timeout)
        future.__del__()
        return Status(code=response.error_code, message=response.reason)

    def set_config(self, parent_key, child_key, value):
        cmd = "set_config {}.{} {}".format(parent_key, child_key, value)
        return self._cmd(cmd)

    def get_config(self, parent_key, child_key):
        cmd = "get_config {}.{}".format(parent_key, child_key)
        return self._cmd(cmd)
