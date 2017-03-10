# region Interchange Words
# import scrabble
# lengthyWordsArray = []
# for word in scrabble.wordList:   # iteration over each word in dictionary
#     if word[:] == word[::-1]:    # word[::x] => x: is the index step for iteration
#         print(word)
# endregion

# region Find letters with no sequenced duplicate in dictionary
# import time
# import string
# import scrabble
# letters = string.ascii_lowercase
# start_time = time.time()
# for letter in letters:
#     exist = False
#     for word in scrabble.wordList:
#         if (letter * 2) in word:
#             exist = True
#             break
#
#     if exist==False:
#         print("There are no English words with a double " + letter)
#
# stop_time = time.time()
#
# print ("The elapsed time is %.1f seconds" % (stop_time - start_time))
# endregion


# region if word has all vowels
import scrabble

vowels = "aeiou"
def has_all_vowels(word):
    for vowel in list(vowels):
        if vowel not in word:
            return False
    return True

for word in scrabble.wordList:
    if has_all_vowels(word):
        print(word)
# endregion









