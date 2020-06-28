import sys


def main():
    cnt = 0
    f_cnt = 0
    input_file = sys.argv[1]
    output_prefix = sys.argv[2]
    chunk_size = int(sys.argv[3])
    f_ov = open(f'{output_prefix}.valid.txt', 'w', encoding='utf-8')
    f_ot = None
    with open(input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            if cnt % 200 == 199:
                f_ov.write(line)
            else:
                if cnt // chunk_size >= f_cnt:
                    f_ot = open(f'{output_prefix}.train.txt.{f_cnt}', 'w', encoding='utf-8')
                    f_cnt += 1
                f_ot.write(line)
            cnt += 1
    f_ov.close()
    f_ot.close()


if __name__ == '__main__':
    main()
