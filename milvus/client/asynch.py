import abc
import threading

from .abstract import TopKQueryResult
from .types import Status


class AbstractFuture:
    @abc.abstractmethod
    def result(self, **kwargs):
        '''Return deserialized result.

        It's a synchronous interface.It will wait executing until
        server response or timeout occur(if specified).

        This API is thread-safe.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def cancel(self):
        '''Cancle gRPC future.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def done(self):
        '''Wait for request done.
        '''
        raise NotImplementedError()


class Future(AbstractFuture):
    def __init__(self, future, done_callback=None):
        self._future = future
        self._done_cb = done_callback
        self._condition = threading.Condition()
        self._done = False
        self._response = None

        self.__init()

    @abc.abstractmethod
    def on_response(self, response):
        raise NotImplementedError()

    def __init(self):
        ''' Register request done callback of gRPC future
        Callback function can be executed in individual subthread of gRPC, so
        there need to notify main thread when callback function finished.
        '''

        def async_done_callback(future):
            with self._condition:
                self._response = future.result()

                # If user specify done callback function, execute it.
                self._done_cb and self._done_cb(*self.on_response(self._response))
                self._done = True
                self._condition.notify_all()

        self._future.add_done_callback(async_done_callback)

    def result(self, **kwargs):
        with self._condition:
            # future not finished. wait callback being called.
            if not self._done:
                to = kwargs.get("timeout", None)
                self._condition.wait(to)

            self._condition.notify_all()

        if kwargs.get("raw", False) is True:
            # just return response object received from gRPC
            return self._response

        return self.on_response(self._response)

    def cancel(self):
        with self._condition:
            if not self._done:
                self._future.cancel()

    def done(self):
        with self._condition:
            if not self._done:
                self._condition.wait()

            self._condition.notify_all()


class SearchFuture(Future):

    def on_response(self, response):
        if response.status.error_code == 0:
            return Status(message='Search successfully!'), TopKQueryResult(response)

        return Status(code=response.status.error_code, message=response.status.reason), None


class InsertFuture(Future):
    def on_response(self, response):
        status = response.status
        if status.error_code == 0:
            return Status(message='Add vectors successfully!'), list(response.vector_id_array)

        return Status(code=status.error_code, message=status.reason), []


class CreateIndexFuture(Future):
    def on_response(self, response):
        status = response.status
        return Status(code=status.error_code, message=status.reason)
