from typing import Tuple

import numpy as np
import torch
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi
from Pinyin2Hanzi import is_pinyin


class Encode:
    # 读取220个音调存入hashmap中
    def __init__(self):
        self.char_map = {' ': 0, "'": 1}
        self.index_map = {0: ' ', 1: "'"}
        with open("../dataset/voice/resource/dict/extra_questions.txt", encoding="utf8") as f:
            i = 2
            for line in f.readlines():
                for ch in line.strip().split(' '):
                    self.char_map[ch] = int(i)
                    self.index_map[int(i)] = ch
                    i += 1

    # 根据字典， 由音调返回index
    def text_to_int(self, text):
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    # 由音调返回相应的index
    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return string


encode = Encode()


class Decode:
    def __init__(self):
        self._hmmparams = DefaultHmmParams()
        # dic 存放 音调转换拼音
        self.dic = {}
        with open("../dataset/voice/resource/dict/pinyin2phone.tone.txt") as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                self.dic[line[1] + line[2]] = line[0]

    def pinyin2chinese(self, labels: list) -> Tuple[str, str]:
        if len(labels) % 2 != 0:
            labels.append('')
        pinyin = []
        # 规范, 两两组合转为拼音
        for i in range(0, len(labels), 2):
            x = self.transform(labels[i] + labels[i + 1])
            while not is_pinyin(x) and i < len(labels)-2:
                i += 1
                x = self.transform(labels[i] + labels[i + 1])
            else:
                if i != len(labels)-2:
                    pinyin.append(x)
        if len(labels) == 0:
            return ''
        # 使用hmm模型和viterbi算法生成中文
        result = viterbi(hmm_params=self._hmmparams, observations=pinyin, path_num=1, log=True)[0]
        result = ''.join(result.path)
        return result, ' '.join(pinyin)

    # 音调到拼音, 并且全部小写
    def transform(self, x: str):
        # change to no digit str
        if x in self.dic:
            # to lower case
            result = self.dic[x].lower()
            # remove digit
            # return result.rstrip(result[-1])
            return result
        else:
            return ''

    # 根据模型输出的tensor输出解码pinyin
    def greed_decode(self, output, labels=None, label_lengths=0, blank_label=0, collapse_repeated=True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []
        targets = []
        for i, args in enumerate(arg_maxes):
            if labels is not None:
                # 根据标签转化为音素
                targets.append(encode.int_to_text(labels[i][:label_lengths[i]].tolist()))
            de = []
            for j, index in enumerate(args):
                index = index.item()
                if index != blank_label:
                    # 删除相连之间重复的
                    if collapse_repeated and j != 0 and index == args[j - 1]:
                        continue
                    de.append(index)
            decodes.append(encode.int_to_text(de))
        return decodes, targets


decode = Decode()


# def avg_wer(wer_scores, combined_ref_len):
#     return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0, n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer
