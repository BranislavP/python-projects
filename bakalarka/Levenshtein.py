import numpy as np


def levenshtein(bad, correct):
    first = [i for i in range(len(correct)+1)]
    second = [0 for i in range(len(correct)+1)]
    i = 0
    while i < len(bad):
        second[0] = i+1
        j = 1
        while j < len(first):
            insert = second[j - 1] + 1
            delete = first[j] + 1
            substitute = first[j - 1] if bad[i] == correct[j - 1] else first[j - 1] + 1
            second[j] = min(insert, delete, substitute)
            j += 1
        temp = first
        first = second
        second = temp
        i += 1
    return first[len(first) - 1]


def weighted_levenshtein(bad, correct, insert_costs=None, delete_costs=None, substitution_costs=None):
    first = [i for i in range(len(correct)+1)]
    second = [0 for i in range(len(correct)+1)]
    i = 0
    while i < len(bad):
        second[0] = i+1
        j = 1
        while j < len(first):
            insert = second[j - 1] + (insert_costs[ord(correct[j - 1])]
                                      if insert_costs is not None and ord(correct[j - 1]) < 400 else 1)
            delete = first[j] + (delete_costs[ord(bad[i])] if delete_costs is not None and ord(bad[i]) < 400 else 1)
            substitute = first[j - 1] if correct[j - 1] == bad[i] else\
                first[j - 1] + (substitution_costs[ord(bad[i])][ord(correct[j - 1])] if substitution_costs is not None
                                and ord(bad[i]) < 400 and ord(correct[j - 1]) < 400 else 1)
            second[j] = min(insert, delete, substitute)
            j += 1
        temp = first
        first = second
        second = temp
        i += 1
    return first[len(first) - 1]


def default_insertion():
    insert = np.ones(400, dtype=np.float64)
    for char in "0123456789":
        insert[ord(char)] = 1.5
    for char in " \t\n":
        insert[ord(char)] = 0
    for char in "\'\"\\/,.-_=:`;()[]{}^!<>":
        insert[ord(char)] = 0.5
    return insert


def default_deletion():
    delete = np.ones(400, dtype=np.float64)
    for char in "0123456789":
        delete[ord(char)] = 1.5
    for char in " \t\n":
        delete[ord(char)] = 0
    for char in "\'\"\\/,.-_=:`;()[]{}^!<>+*|%&$#@":
        delete[ord(char)] = 0.5
    return delete


def default_substitution():
    substitution = np.ones((400, 400), dtype=np.float64)
    for i in range(0,400):
        for char in "0123456789":
            if ord(char) != i:
                substitution[i][ord(char)] = 1.75
    for char in "0123456789":
        for c in "0123456789":
            if char != c:
                substitution[ord(char)][ord(c)] = 2.5
                substitution[ord(c)][ord(char)] = 2.5
    substitution[ord('.')][ord(',')] = 0.25
    substitution[ord(',')][ord('.')] = 0.25
    substitution[ord(':')][ord(';')] = 0.25
    substitution[ord(';')][ord(':')] = 0.25
    substitution[ord(':')][ord('=')] = 0.25
    substitution[ord('=')][ord(':')] = 0.25
    substitution[ord('\'')][ord(',')] = 0.25
    substitution[ord(',')][ord('\'')] = 0.25
    substitution[ord('9')][ord('g')] = 0.75
    substitution[ord('1')][ord('l')] = 0.75
    substitution[ord('1')][ord('i')] = 0.75
    substitution[ord('1')][ord('í')] = 0.75
    substitution[ord('1')][ord('I')] = 0.75
    substitution[ord('1')][ord('|')] = 0.75
    substitution[ord('0')][ord('O')] = 0.75
    substitution[ord('0')][ord('o')] = 0.75
    substitution[ord('0')][ord('Q')] = 0.75
    similiar = ["aáä", "eéě", "cčCČoóOÓô", "DOQ", "yvuúý", "iIlíľĺ", "tť", "TŤ", "AÁ", "EÉ", "UÚV", "IÍ", "pP", "sS",
                "dď", "LĽĹ", "zžZŽ", "xX", "nň", "NŇ"]
    for string in similiar:
        for ch in string:
            for c in string:
                substitution[ord(c)][ord(ch)] = 0.75
    return substitution


def default_weights():
    return default_insertion(), default_deletion(), default_substitution()

