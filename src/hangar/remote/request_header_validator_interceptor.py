# Copyright 2017 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Interceptor that ensures a specific header is present."""

import grpc
from os.path import split

SERVICE_METHOD_TYPES = {
    'GetClientConfig': 'uu',
    'FetchBranchRecord': 'uu',
    'FetchData': 'ss',
    'FetchLabel': 'uu',
    'FetchCommit': 'us',
    'FetchSchema': 'uu',
    'PushBranchRecord': 'uu',
    'PushData': 'su',
    'PushLabel': 'uu',
    'PushCommit': 'su',
    'PushSchema': 'uu',
    'FetchFindMissingCommits': 'uu',
    'FetchFindMissingHashRecords': 'ss',
    'FetchFindMissingLabels': 'ss',
    'FetchFindMissingSchemas': 'uu',
    'PushFindMissingCommits': 'uu',
    'PushFindMissingHashRecords': 'ss',
    'PushFindMissingLabels': 'ss',
    'PushFindMissingSchemas': 'uu',
}


def _unary_unary_rpc_terminator(code, details):

    def terminate(ignored_request, context):
        context.abort(code, details)

    return grpc.unary_unary_rpc_method_handler(terminate)


def _unary_stream_rpc_terminator(code, details):

    def terminate(ignored_request, context):
        context.abort(code, details)

    return grpc.unary_stream_rpc_method_handler(terminate)


def _stream_unary_rpc_terminator(code, details):

    def terminate(ignored_request, context):
        context.abort(code, details)

    return grpc.unary_stream_rpc_method_handler(terminate)


def _stream_stream_rpc_terminator(code, details):

    def terminate(ignored_request, context):
        context.abort(code, details)

    return grpc.stream_stream_rpc_method_handler(terminate)


def _select_rpc_terminator(intercepted_method):
    method_type = SERVICE_METHOD_TYPES[intercepted_method]
    if method_type == 'uu':
        return _unary_unary_rpc_terminator
    elif method_type == 'su':
        return _stream_unary_rpc_terminator
    elif method_type == 'us':
        return _unary_stream_rpc_terminator
    elif method_type == 'ss':
        return _stream_stream_rpc_terminator
    else:
        raise ValueError(f'unknown method type: {method_type} for service: {intercepted_method}')


class RequestHeaderValidatorInterceptor(grpc.ServerInterceptor):

    def __init__(self, push_restricted, header, value, code, details):
        self._push_restricted = push_restricted
        self._header = header
        self._value = value
        self._code = code
        self._details = details
        # self._terminator = _select_rpc_terminator(method)(code, details)

    def intercept_service(self, continuation, handler_call_details):
        _, intercepted_method = split(handler_call_details.method)
        print(f'intercepted method: {intercepted_method}')
        # print(self.__dict__)

        if (intercepted_method.startswith('Push') is True) and (self._push_restricted is True):
            if (self._header, self._value) in handler_call_details.invocation_metadata:
                return continuation(handler_call_details)
            else:
                return _select_rpc_terminator(intercepted_method)(self._code, self._details)

        else:
            return continuation(handler_call_details)
            # return _select_rpc_terminator(intercepted_method)(self._code, self._details)
