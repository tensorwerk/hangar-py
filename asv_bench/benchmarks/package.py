import subprocess
import sys


class TimeImport(object):

    def time_import(self):
        if sys.version_info >= (3, 7):
            # on py37+ we the "-X importtime" usage gives us a more precise
            #  measurement of the import time we actually care about,
            #  without the subprocess or interpreter overhead
            cmd = [sys.executable, "-X", "importtime", "-c", "import hangar"]
            p = subprocess.run(cmd, stderr=subprocess.PIPE)

            line = p.stderr.splitlines()[-1]
            field = line.split(b"|")[-2].strip()
            total = int(field)  # microseconds
            return total

        cmd = [sys.executable, "-c", "import hangar"]
        subprocess.run(cmd, stderr=subprocess.PIPE)