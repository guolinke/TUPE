import sys

max_len = int(sys.argv[1])

for line in sys.stdin:
    lst = line.strip().split(' ')
    if len(lst) <= max_len:
        sys.stdout.write(line)
    else:
        sys.stdout.write(" ".join(lst[:max_len]) + '\n')
