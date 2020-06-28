#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <set>

using namespace std;

void die(const char *msg) {
  cerr << msg << endl;
  exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
  ifstream fin;
  string linebuf;
  unordered_map<string, uint64_t> wordfreq;
  cerr << "corpus file: " << argv[1] << endl;
  cerr << "unigram file: " << argv[2] << endl;
  cerr << "filtered corpus file: " << argv[3] << endl;
  cerr << "cutoff ratio: " << argv[4] << endl;

  auto cutoff = strtod(argv[4], nullptr);

  if (cutoff <= 0.0 || cutoff > 1.0 || isinf(cutoff) || isnan(cutoff)) {
    die("invalid cutoff");
  }

  fin.open(argv[1]);
  if (!fin) {
    die("cannot open input file");
  }

  cerr << "processing unigram..." << endl;

  auto proc_words = [&](auto& fin, auto fn) {
    while(getline(fin, linebuf)) {
      auto pidx = 0;
      auto len = linebuf.length();
      if (len == 0) {
        continue;
      }
      do {
        auto sidx = linebuf.find(' ', pidx);
        if (sidx == linebuf.npos) {
          sidx = len;
        }
        auto word = linebuf.substr(pidx, sidx - pidx);
        fn(pidx == 0, sidx >= len, word);
        pidx = sidx + 1;
      } while(pidx < len);
    }
  };

  proc_words(fin, [&](auto _, auto __, const auto& word) {
      const auto [it, ins] = wordfreq.insert({word, 1});
      if (!ins) {
        ++it->second;
      }
  });

  cerr << "sorting by frequency..." << endl;
  auto compFunctor = [](const pair<string, uint64_t> &a,
                        const pair<string, uint64_t> &b) {
    return a.second > b.second || (a.second == b.second && a.first < b.first);
  };

  set<pair<string, uint64_t>, decltype(compFunctor)> sorted_unigram_freq(
      wordfreq.begin(), wordfreq.end(), compFunctor);
  wordfreq.clear();

  FILE* fp = fopen(argv[2], "w");
  if (!fp) {
    die("cannot open unigram file");
  }

  cerr << "saving unigram data..." << endl;
  for (const auto &[k, v] : sorted_unigram_freq) {
    fprintf(fp, "%s %lu\n", k.data(), v);
  }
  fclose(fp);

  cerr << "unigram built. size = " << sorted_unigram_freq.size() << endl;

  auto cutoff_offset = min(sorted_unigram_freq.size(), (uint64_t)(sorted_unigram_freq.size() * cutoff));
  auto cutoff_it = sorted_unigram_freq.begin();
  for(auto i = 0; i < cutoff_offset; ++i) {
    ++cutoff_it;
  }
  sorted_unigram_freq.erase(cutoff_it, sorted_unigram_freq.end());

  cerr << "unigram filtered. size = " << sorted_unigram_freq.size() << endl;

  // put it back to the dict...
  wordfreq.insert(sorted_unigram_freq.begin(), sorted_unigram_freq.end());
  sorted_unigram_freq.clear();

  fin.seekg(0);
  fp = fopen(argv[3], "w");
  if (!fp) {
    die("cannot open filtered corpus file");
  }

  cerr << "wordfreq-filtering corpus data" << endl;
  fin.close();
  fin.open(argv[1]);
  if (!fin) {
    die("cannot open input file");
  }

  proc_words(fin, [&](auto first, auto last, const auto& word) {
    if (!first) fputc(' ', fp);
    if(wordfreq.end() == wordfreq.find(word)) {
      fputs("<unk>", fp);
    } else {
      fputs(word.data(), fp);
    }
    if (last) fputc('\n', fp);
  });

  fclose(fp);
  cerr << "corpus wordfreq filter complete." << endl;

  return EXIT_SUCCESS;
}
