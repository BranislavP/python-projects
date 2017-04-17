# This Python file uses the following encoding: utf-8

# Section 2. Higher order functions and list comprehension
#
# According to Wikipedia, high-order-functions are functions that do at least one
# of the following:
#
#   Take one or more functions as input.
#   Output a function.
#
# In python, there are several examples of such functions, some of them are:
#
#   filter( function, iterable )
#   map( function, iterable )
#   reduce( function, iterable )
#   max( iterable )
#   sort( iterable )

from functools import reduce


# 26. Largest Number.
# Given a list of numbers, this function returns the largest one,
# it does it using `reduce` function.
def max_in_list_v1( numbers ):
    return reduce(lambda largest, current: largest if (largest > current) else current,numbers)


# 27. Map strings
# Maps a list of words into a list of integers representing each
# word length.
# It uses the `map()` higher-order-function.
def map_words_v1( words ):
    return list(map(lambda x: len(x), words))


# 27.1 Map strings.
# Does the same as #27 but uses `List comprehensions`.
# See http://www.secnetix.de/olli/Python/list_comprehensions.hawk
def map_words_v2( words ):
    return [len(x) for x in words]


# 28. Find longest word.
# Using high-order-functions, this function receives a list of words,
# and return the longest one.
def find_longest_word_advanced( words ):
    return len(reduce(lambda lar, current: lar if (len(lar) > len(current)) else current, words))


# 29. Filter Long Words.
# Takes a list of words and an integer and then return
# an array with all the words that are longer in length that
# the integer passed.
def filter_long_words_advanced( words, x ):
    return list(filter(lambda y: len(y) >= x, words))


# 30. Translates a list of words from English into Swedish.
# There are a fixed amount of words available for translatation.
# Uses `map()`.
# Sample:
#
#   ( [ 'happy', 'new', 'year' ] ) -> [ 'happy', 'nya', 'år' ]
#
def translate_words( words ):

    d = { 'all': 'alla', 'and': 'och', 'best': 'bäst', 'birthday': 'födelsedag', 'cake': 'kaka', 'celebrate': 'fira', 'christmas': 'jul',
        'come': 'komma', 'dreams': 'drömmar', 'for': 'för', 'good': 'bra', 'happy': 'happy', 'hope': 'hopp', 'inviting': 'inbjudande', 'is': 'är',
        'merry': 'merry', 'new': 'nya',  'thanks': 'tack', 'the': 'den', 'to': 'till', 'true': 'sant', 'us': 'oss', 'we': 'vi',
        'wishes': 'önskemål', 'year': 'år', 'your': 'din' }

    return list(map(lambda x: d[x], words))


# 31. Map.
# `map()` is a python build-in function which receive a function `f`
# and a list as parameters, and calls the function `f` on every element
# from the list.
# The following function is an implementation of the functionality that
# `map()` accomplishes.
def map_v1( fn, iterable ):
    return [fn(x) for x in iterable]


# 31.1 Filter.
# As well as `map()`, `filter()` is a python built-in higer-order function,
# which get passed a function `f` and a list, and applies `f` to each of the elementf
# in the list, and return a list with only the elements that when passed to `f`
# made `f` return true.
def filter_v1( fn, iterable ):
    a = list()
    for x in iterable:
        if fn(x):
            a.append(x)
    return a


# 31.2 Reduce.
# Reduce is also a built-in higer-priority function in python.
# It receives two parameters, the first, a function `f` which receives two
# arguments: `a`, `b`, and the second an iterable `iterable` which can ba a list.
# What `reduce()` does, is calling `f` passing it as parameters the first and the second
# elements in the list, after this, reduce captures the return value of the function
# and passes it to `f` as `a`, and as `b` passes the third element of the list,
# `reduce()` captures the return value and passes it to `f` as `a` and so on, until
# there are no more elements in the list.
#
#   nums = [ 1, 2, 3, 4, 5 ]
#   sum = lambda x, y: x + y
#   reduce_v1( sum, nums )
#  -> 15
#
def reduce_v1( fn, iterable ):
    a = fn(iterable[0], iterable[1])
    for i in range(2, len(iterable)):
        a = fn(a, iterable[i])
    return a

