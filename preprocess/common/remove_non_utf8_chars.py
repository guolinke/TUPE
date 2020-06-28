import io
import sys

for line in io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore'):
    sys.stdout.write(line)
