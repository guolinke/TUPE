import sys
import re

for line in sys.stdin:
    re.sub(r" n't\b", "n't", line)
    re.sub(r" 's\b", "'s", line)
    sys.stdout.write(line)
