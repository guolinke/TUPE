import re
import string
import sys
from collections import Counter

def post_cleanup(line):
    line = re.sub(r'\\', ' ', line)  # remove all backslashes
    line = re.sub(r'\s\s+', ' ', line) # remove all redundant spaces
    line = re.sub(r'\.\.\.', ' ... ', line)
    line = re.sub(r'\.\.', ' .. ', line)
    newline = ''
    for w in line.split():
        ls = w.lstrip(string.punctuation)
        rs = ls.rstrip(string.punctuation)
        lw = len(w)
        lstart = lw - len(ls)
        rstart = lstart + len(rs)
        for i in range(lstart):
            newline += ' ' + w[i]
        if rs:
            newline += ' ' + rs
        for i in range(rstart, lw):
            newline += ' ' + w[i]
    return newline.lstrip()


def main():
    # buffer for one output line concatenated from input lines
    for line in sys.stdin:
        newline = line.strip().lower()
        newline = post_cleanup(newline)
        sys.stdout.write(newline + '\n')

if __name__ == '__main__':
    main()
