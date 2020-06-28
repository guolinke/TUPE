import sys

for line in sys.stdin:
    l = len(line)
    if int(sys.argv[1]) <= l <= int(sys.argv[2]):
        sys.stdout.write(line)
