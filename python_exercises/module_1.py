# Section 1. Very simple exercises
#
# This selection of exercises is intended for developers
# to get a basic understanding of logical operators and loops in Python
#

import re


# 1. Max of two numbers.
def max_num( a, b ):
    return a if a > b else b


# 2. Max of three numbers.
def max_of_three( a, b, c ):
    return max([a, b, c])


# 3. Calculates the length of a string.
def str_len( string ):
    return len(string)


# 4. Returns whether the passed letter is a vowel.
def is_vowel( letter ):
    return letter in "aeiouy"


# 5. Translates an English frase into `Robbers language`.
# Sample:
#
#   This is fun
#   Tothohisos isos fofunon
#
def translate( string ):
    result = []
    vowels = "aeiouy"
    for char in string:
        result.append(char)
        if char not in vowels and char.isalpha():
            result.append('o')
            result.append(char.lower())
    return ''.join(result)


# 6. Sum.
# Sums all the numbers in a list.
def sum( items ):
    result = 0
    for item in items:
        result += item
    return result


# 6.1. Multiply.
# Multiplies all the items in a list.
def multiply( items ):
    result = 1
    for item in items:
        result *= item
    return result


# 7. Reverse.
# Reverses a string.
# 'I am testing' -> 'gnitset ma I'
def reverse( string ):
    return string[::-1]


# 8. Is palindrome.
# Checks whether a string is palindrome.
# 'radar' > reversed : 'radar'
def is_palindrome( string ):
    return string == string[::-1]


# 9. Is member.
# Checks whether a value x is contained in a group of values.
#   1 -> [ 2, 1, 0 ] : True
def is_member( x, group ):
    return x in group


# 10. Overlapping.
# Checks whether two lists have at least one number in common
def overlapping( a, b ):
    return set(a) & set(b)


# 11. Generate n chars.
# Generates `n` number of characters of the given one.
#
#   generate_n_chars( 5, 'n' )
#   -> nnnnn
#
def generate_n_chars( times, char ):
    return char*times


# 12. Historigram.
# Takes a list of integers and prints a historigram of it.
#   historigram( [ 1, 2, 3 ] ) ->
#       *
#       **
#       ***
#
def historigram( items ):
    for item in items:
        print(item * "*")


# 13. Max in list.
# Gets the larges number in a list of numbers.
def max_in_list( list ):
    return max(list)


# 14. Map words to numbers.
# Gets a list of words and returns a list of integers
# representing the length of each word.
#
#   [ 'one', 'two', 'three' ] -> [ 3, 3, 5 ]
#
def map_words( words ):
    return [len(item) for item in words]


# 15. Find longest wors.
# Receives a list of words and returns the length
# of the longest one.
#
#   [ 'one', 'two', 'three', 'four' ] -> 5
#
def longest_word( words ):
    return max([len(item) for item in words])


# 16. Filter long words.
# Receives a list of words and an integer `n` and returns
# a list of the words that are longer than n.
def filter_long_words( words, x ):
    return [word for word in words if len(word) >= x]


# 17. Version of palindrome that ignores punctuation, capitalization and
# spaces, so that a larger range of frases can be clasified as palindromes.
#
#   ( "Dammit, I'm mad!" ) -> is palindrome
#
def is_palindrome_advanced( string ):
    string = re.sub('[^a-zA-Z0-9]', '', string.lower())
    return string == string[::-1]


# 18. Is pangram.
# Checks whether a phrase is pangram, that is, if
# it contains all the letters of the alphabet.
def is_pangram( phrase ):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for char in alphabet:
        if char not in phrase:
            return 0
    return 1


# 19. 99 Bottles of beer.
# 99 Bottles of beer is a traditional song in the United States and Canada.
# It has a very repetitive lyrics and it is popular to sing it in very long trips.
# The lyrics of the song are as follows.
#
#   99 bottles of beer in the wall, 99 bottles of beer.
#   Take one down, pass it arrown, 98 bottles of beer.
#
# The song is repeated having one less bottle each time until there are no more
# bottles to count.
#
def sing_99_bottles_of_beer():
    i = 99
    while i > 0:
        print("%d bottles of beer in the wall, %d bottles of beer.\nTake one down, pass it around, %d bottles of beer."
              % (i, i, i-1))
        i -= 1


# 20. Note: exercise number 20 is the same as exercise # 30


# 21. Character frequency.
# Counts how many characters of the same letter there are in
# a string.
#
#   ( 'aabbccddddd' ) -> { 'a': 2, 'b': 2, 'c': 2, d: 5 }
#
def char_freq( string ):
    dict = {}
    for char in string:
        dict[char] = string.count(char, 0, len(string))
    return dict


# 22. ROT-13: Encrypt.
# Encrypts a string in ROT-13.
#
#   rot_13_encrypt( 'Caesar cipher? I much prefer Caesar salad!' ) ->
#   Pnrfne pvcure? V zhpu cersre Pnrfne fnynq!
#
def rot_13_encrypt( string ):
    upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lower = upper.lower()
    result = []
    for char in string:
        if char in lower:
            result.append(lower[(lower.index(char) + 13) % len(lower)])
        elif char in upper:
            result.append(upper[(upper.index(char) + 13) % len(upper)])
        else:
            result.append(char)
    return ''.join(result)

# 22.1 ROT-13: Decrypt.
#
#   rot_13_decrypt( 'Pnrfne pvcure? V zhpu cersre Pnrfne fnynq!' ) ->
#   Caesar cipher? I much prefer Caesar salad!
#
# Since we're dealing with offset 13 it means that decrypting a string
# can be accomplished with the encrypt function given that the alphabet contains
# 26 letters.
def rot_13_decrypt( string ):
   return rot_13_encrypt(string)


# 23. Correct.
# Takes a string and sees that 1) two or more occurences of a space
# are compressed into one. 2) Adds a space betweet a letter and a period
# if they have not space.
#
#   correct( 'This   is  very funny  and    cool.Indeed!' )
#   -> This is very funny and cool. Indeed!
#
def correct( string ):
    pom = re.sub('\.', '. ', string)
    pom = re.sub(' +', ' ', pom)
    return pom.strip()



# 24. Make third Form.
# Takes a singular verb and makes it third person.
#
#   ( 'run' ) -> 'runs'
#   ( 'Brush' ) -> 'brushes'
#
def make_3d_form( verb ):
    consonants = "bcdfghjklmnpqrstvwxyz"
    ends = ('s', 'x', 'z', "ch", "sh")
    irregulars = {"woman": "women", "man": "men", "child": "children", "tooth": "teeth", "foot": "feet",
                  "person": "people", "leaf": "leaves", "mouse": "mice", "goose": "geese", "half": "halves",
                  "knife": "knives", "wife": "wives", "life": "lives", "elf": "elves", "loaf": "loaves",
                  "potato": "potatoes", "tomato": "tomatoes", "cactus": "cacti", "focus": "foci", "fungus": "fungi",
                  "nucleus": "nuclei", "syllabus": "syllabi", "analysis": "analyses", "diagnosis": "diagnoses",
                  "oasis": "oases", "thesis": "theses", "crisis": "crises", "phenomenon": "phenomena",
                  "criterion": "criteria", "datum": "data", "sheep": "sheep", "fish": "fish",
                  "deer": "deer", "species": "species", "aircraft": "aircraft"}
    if verb in irregulars.keys():
        return irregulars[verb]
    elif verb[len(verb) - 1] in ends or verb[len(verb) - 2:] in ends:
        return verb + "es"
    elif verb[len(verb) - 1] == 'y' and verb[len(verb) - 2] in consonants:
        return verb[:len(verb) - 1] + "ies"
    else:
        return verb + "s"


# 25. Make `ing` form.
# Given an infinite verb this function returns the
# present participle of it.
#
#   ( 'go' ) -> 'going'
#   ( 'sleep' ) -> 'sleep'
#
def make_ing_form( verb ):
    if verb[len(verb) - 2:] == "ie":
        return verb[:len(verb) - 2] + "ying"
    elif verb[len(verb) - 1] == 'e':
        return verb[:len(verb) - 1] + "ing"
    elif verb[len(verb) - 1] == 'n':
        return verb + "ning"
    else:
        return verb + "ing"

