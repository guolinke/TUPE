import re
import sys


def pre_cleanup(line):
    line = line.replace('\t', ' ')  # replace tab with spaces
    line = ' '.join(line.strip().split())  # remove redundant spaces
    line = re.sub(r'\.{4,}', '...', line)  # remove extra dots
    line = line.replace('<<', '«').replace('>>', '»')  # group << together
    line = re.sub(' (,:\.\)\]»)', r'\1', line)  # remove space before >>
    line = re.sub('(\[\(«) ', r'\1', line)  # remove space after <<
    line = line.replace(',,', ',').replace(',.', '.')  # remove redundant punctuations
    line = re.sub(r' \*([^\s])', r' \1', line)  # remove redundant asterisks
    return ' '.join(line.strip().split())  # remove redundant spaces


def main():
    for line in sys.stdin:
        line = pre_cleanup(line)
        sys.stdout.write(line + '\n')


if __name__ == '__main__':
    main()
