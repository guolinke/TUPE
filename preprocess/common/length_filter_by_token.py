import sys

for line in sys.stdin:
    l = len(line.strip().split(' '))
    if int(sys.argv[1]) <= l <= int(sys.argv[2]):
        sys.stdout.write(line)
