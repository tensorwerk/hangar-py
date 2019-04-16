import os
from os import remove
from shutil import move

from grpc_tools import protoc

protoPath = os.path.dirname(__file__)
os.environ.putenv('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'cpp')

# ------------------------ hangar service -------------------------------------

hangar_service_path = os.path.join(protoPath, 'hangar_service.proto')
# generates hangar service protobuf for python
protoc.main((
    '',
    '-I.',
    '--python_out=.',
    '--grpc_python_out=.',
    hangar_service_path,
))

'''
Because python3 requires explicit relative imports (which is not yet supported
in the Google protoc compiler), we have to replace the 'import foo_grpc' line
with the 'from . import foo' line in the generated grpc code.
'''

hangar_service_grpc_path_orig = os.path.join(protoPath, 'hangar_service_pb2_grpc.py')
hangar_service_grpc_path_old = os.path.join(protoPath, 'hangar_service_pb2_grpc.py.old')
move(hangar_service_grpc_path_orig, hangar_service_grpc_path_old)
with open(hangar_service_grpc_path_orig, 'w') as new_file:
    with open(hangar_service_grpc_path_old, 'r+') as old_file:
        for old_line in old_file:
            if old_line == 'import hangar_service_pb2 as hangar__service__pb2\n':
                newline = old_line.replace('import', 'from . import')
            else:
                newline = old_line
            new_file.writelines(newline)
remove(hangar_service_grpc_path_old)