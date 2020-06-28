import re
import string
import sys


#  extract_patterns does the following things:
#  - special patterns are extracted
#    - email addresses
#    - urls
#    - files
#  - tokenize by hyphens in words, light-hearted etc.
#
#  !! requires filter_and_cleanup_lines.py.
#  !! requires all lowercase input.
def extract_patterns(line):
    email = r'^([a-z0-9_\-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-z0-9\-]+\.)+))([a-z]{2,4}|[0-9]{1,3})(\]?)$'
    # ref: https://www.w3schools.com/php/php_form_url_email.asp
    url1 = r'^(?:(?:https?|ftp):\/\/|www\.)[-a-z0-9+&@#\/%?=~_|!:,.;]*[-a-z0-9+&@#\/%=~_|]$'
    # simple fooo-bar.com cases without the prefix
    url2 = r'^[^$s]{3}[^$s]*\.(?:com|net|org|edu|gov)$'
    # file: prefix len >=5, suffix given.
    file = r'^[a-z_-][a-z0-9_-]{4}[a-z0-9_-]*\.(?:pdf|mp4|mp3|doc|xls|docx|ppt|pptx|wav|wma|csv|tsv|cpp|py|bat|reg|png|jpg|mov|avi|gif|rtf|txt|bmp|mid)$'
    newline = ''
    for w in line.split():
        w = re.sub(url1, '<url>', w)
        w = re.sub(url2, '<url>', w)
        w = re.sub(email, '<email>', w)
        w = re.sub(file, '<file>', w)
        w = ' - '.join(w.split('-'))
        newline += ' ' + w
    return newline.lstrip()


def main():
    for line in sys.stdin:
        line = extract_patterns(line)
        sys.stdout.write(line + '\n')

if __name__ == '__main__':
    main()
