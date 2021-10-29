import os


class Encode:
    def __init__(self):
        self.char_map = {" ": 0}
        self.index_map = {0: " "}
        with open("../dataset/voice/resource/dict/extra_questions.txt") as f:
            i = 1
            for line in f.readlines():
                for ch in line.strip().split(' '):
                    self.char_map[ch] = int(i)
                    self.index_map[int(i)] = ch
                    i += 1

    def text_to_int(self, text):
        int_sequence = []
        for c in text:
            ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return string


class Decode:
    def __init__(self):
        self.dic = {}
        with open("../dataset/voice/resource/dict/lexicon.txt") as f:
            i = 0
            for line in f.readlines():
                line = line.strip().split(' ')
                value = line[0]
                line.pop(0)
                key = "".join(line)
                self.dic[key] = value

test = Encode()