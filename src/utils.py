from collections import Counter

def most_common_value(sequence):
    counter = Counter(sequence)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None

def special_characters_prediction(sentence, character):
    if sentence:
        character = 'AW' if (sentence[-1] in ['A', 'AW'] and character == 'Munguoc') else character
        character = 'AA' if (sentence[-1] in ['A', 'AA'] and character == 'Mu') else character
        character = 'EE' if (sentence[-1] in ['E', 'EE'] and character == 'Mu') else character
        character = 'UW' if (sentence[-1] in ['U', 'UW'] and character == 'Rau') else character
        character = 'OW' if (sentence[-1] in ['O', 'OW'] and character == 'Rau') else character
        character = 'OO' if (sentence[-1] in ['O', 'OO'] and character == 'Mu') else character
    return character

# Ánh xạ ký tự đặc biệt sang dạng mong muốn
char_display_map = {
    "EE": "Ê",
    "AA": "Â",
    "OW": "Ơ",
    "AW": "Ă",
    "UW": "Ư",
    "OO": "Ô",
    "DD": "Đ"
}
