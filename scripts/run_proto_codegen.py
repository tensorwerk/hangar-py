import os
from os import remove
from shutil import move

from grpc_tools import protoc


# ------------------------- output locations ----------------------------------


toolsPath = os.path.dirname(__file__)
srcPath = os.path.normpath(os.path.join(toolsPath, os.path.pardir, 'src'))

hangarProtoDir = os.path.join(srcPath, 'hangar', 'remote')
hangarProtoPath = os.path.join(hangarProtoDir, 'hangar_service.proto')
if not os.path.isfile(hangarProtoPath):
    raise FileNotFoundError(f'Cannot access hangar_service.proto at: {hangarProtoPath}')

# ------------------------ hangar service -------------------------------------

os.environ.putenv('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'cpp')
# generates hangar service protobuf for python
protoc.main((
    '',
    f'-I{hangarProtoDir}',
    f'--python_out={hangarProtoDir}',
    f'--grpc_python_out={hangarProtoDir}',
    f'--mypy_out={hangarProtoDir}',
    hangarProtoPath,
))

"""
Because python3 requires explicit relative imports (which is not yet supported
in the Google protoc compiler), we have to replace the 'import foo_grpc' line
with the 'from . import foo' line in the generated grpc code.
"""

hangar_service_grpc_path_orig = os.path.join(hangarProtoDir, 'hangar_service_pb2_grpc.py')
hangar_service_grpc_path_old = os.path.join(hangarProtoDir, 'hangar_service_pb2_grpc.py.old')
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