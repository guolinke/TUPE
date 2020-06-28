import re
import string
import sys
from collections import Counter

#  globals
pending_tail = string.ascii_letters + string.digits + ','


#  Filter invalid lines
def is_valid(line):
    l = len(line)
    if l > 1000000 or l < 50:
        return False
    count = Counter(line)
    alpha_cnt = sum(count[ch] for ch in string.ascii_letters)
    if alpha_cnt < 50 or alpha_cnt / l < 0.7:
        return False
    # if count['/'] / l > 0.05:  # filter hyperlinks
        # return False
    if count['\\'] / l > 0.05:  # filter latex math equations
        return False
    if count['|'] / l > 0.05 or line[0] == '|':  # filter remaining tables
        return False
    return True


#  post_cleanup does the following things:
#  - remove all backslashes
#  - normalize and remove redundant spaces (including \t, etc.)
#  - tokenize by spaces, and start/end puncts, normalize each word
#    - puncts in the middle are regarded as a part of the word, for example,
#      1.23, y'all, etc.
#  - replace '...' with ' ... '
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


def write_output(line):
    if is_valid(line):
        line = post_cleanup(line)
        sys.stdout.write(line + '\n')

#  Two lines can be concatenated if:
#  - both lines are not empty
#  - line1 ends with "pending_tail" (letters, digits, and ',')
#  - line2 begins with lower-cased letters
def check_concat(line1, line2):
    global pending_tail
    if len(line1) == 0 or len(line2) == 0:
        return False
    return (line1[-1] in pending_tail) and (line2[0] in string.ascii_lowercase)


def main():
    # buffer for one output line concatenated from input lines
    buf = []
    for line in sys.stdin:
        line = line.strip().lower()
        if buf and (not check_concat(buf[-1], line)):
            write_output(' '.join(buf))
            buf = []
        buf.append(line)
    if buf:
        write_output(' '.join(buf))


if __name__ == '__main__':
    main()
