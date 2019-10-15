"""Interceptor that adds headers to outgoing requests

Portions of this code have been taken and modified from the "gRPC" project.

URL:      https://github.com/grpc/grpc/
File:     examples/python/interceptors/default_value/header_manipulator_client_interceptor.py
Commit:   87cd994b0477e98c976e7b321b3c1f52666ab5e0
Accessed: 23 APR 2019

gRPC License
-------------------------------------------------------------------------------
Copyright 2017 gRPC authors.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""
from collections import namedtuple

import grpc


class _GenericClientInterceptor(
        grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor,
        grpc.StreamUnaryClientInterceptor, grpc.StreamStreamClientInterceptor):
    """Base class for interceptors that operate on all RPC types."""

    def __init__(self, interceptor_function):
        self._fn = interceptor_function

    def intercept_unary_unary(self, continuation, client_call_details, request):
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, iter((request,)), False, False)
        response = continuation(new_details, next(new_request_iterator))
        return postprocess(response) if postprocess else response

    def intercept_unary_stream(self, continuation, client_call_details,
                               request):
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, iter((request,)), False, True)
        response_it = continuation(new_details, next(new_request_iterator))
        return postprocess(response_it) if postprocess else response_it

    def intercept_stream_unary(self, continuation, client_call_details,
                               request_iterator):
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, request_iterator, True, False)
        response = continuation(new_details, new_request_iterator)
        return postprocess(response) if postprocess else response

    def intercept_stream_stream(self, continuation, client_call_details,
                                request_iterator):
        new_details, new_request_iterator, postprocess = self._fn(
            client_call_details, request_iterator, True, True)
        response_it = continuation(new_details, new_request_iterator)
        return postprocess(response_it) if postprocess else response_it


def create_client_interceptor(intercept_call):
    return _GenericClientInterceptor(intercept_call)


class _ClientCallDetails(
        namedtuple(
            typename='_ClientCallDetails',
            field_names=('method', 'timeout', 'metadata', 'credentials')),
        grpc.ClientCallDetails):
    pass


def header_adder_interceptor(header, value):
    """Interceptor that adds headers to outgoing requests."""

    def intercept_call(client_call_details, request_iterator, request_streaming,
                       response_streaming):
        metadata = []
        if client_call_details.metadata is not None:
            metadata = list(client_call_details.metadata)

        if (header != '') and (value != ''):
            metadata.append((header, value))
        client_call_details = _ClientCallDetails(
            client_call_details.method,
            client_call_details.timeout,
            metadata,
            client_call_details.credentials)
        return client_call_details, request_iterator, None

    return create_client_interceptor(intercept_call)