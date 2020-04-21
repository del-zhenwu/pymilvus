class ParamError(ValueError):
    """
    Param of interface is illegal
    """


class ConnectError(ValueError):
    """
    Connect server failed
    """


class NotConnectError(ConnectError):
    """
    Disconnect error
    """


class RepeatingConnectError(ConnectError):
    """
    Try to connect repeatedly
    """


class ConnectionPoolError(ConnectError):
    """
    Waiting timeout error
    """


class DeprecatedError(AttributeError):
    """
    API is deprecated
    """


class FutureTimeoutError(TimeoutError):
    """
    Future timeout
    """


class ResponseError(ValueError):
    """
    Server response error
    """

    def __init__(self, code, message):
        self._code = code
        self._message = message

    def __str__(self):
        return "ResponseError(code={}, message={})".format(self._code, self._message)
