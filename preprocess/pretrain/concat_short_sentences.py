import sys


def score(line1, line2):  # the smaller the more likely to be concat
    s = len(line1) + len(line2)
    if s > 250:
        return 9999999
    if line1[-1] in ['.', '"', '!', '?']:
        s += 5
    return s


def main():
    buf = []
    for line in sys.stdin:
        words = line.strip().split()
        if len(words) == 0:
            while True:
                if min([len(sent) for sent in buf]) >= 5:
                    break
                mi, best = 9999999, None
                for i in range(len(buf) - 1):
                    s = score(buf[i], buf[i + 1])
                    if s < mi:
                        mi = s
                        best = i
                if best is None:
                    break
                buf[best] = buf[best] + buf[best + 1]
                buf.pop(best + 1)
            sys.stdout.write(''.join(' '.join(sent) + '\n' for sent in buf) + '\n')
            buf = []
        else:
            buf.append(words)


if __name__ == '__main__':
    main()
