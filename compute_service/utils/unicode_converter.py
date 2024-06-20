import re
import warnings


def sinhala_to_singlish(text):

    sinhala_pattern = re.compile(r'[\u0D80-\u0DFF]')
    if not bool(sinhala_pattern.search(text)):
        warnings.warn("Given text has at least one non sinhala letters.")

    def __get_lexical_result(grapheme):
        return __sinhala_lexical.get(grapheme, [grapheme, "common"])

    graphemes = list(text)

    singlish_text = ''

    for grapheme in graphemes:
        lexical_result = __get_lexical_result(grapheme)
        singlish_grapheme = lexical_result[0]
        grapheme_type = lexical_result[1]

        if grapheme_type == "vowel" or \
            grapheme_type == "consonant" or \
            grapheme_type == "common":
            singlish_text+=singlish_grapheme
        elif grapheme_type == "glyph":
            singlish_text = singlish_text[:-1] + singlish_grapheme
        
    return singlish_text


__sinhala_lexical = {
    "අ": ["a","vowel"],
    "ආ": ["aa","vowel"],
    "ඇ": ["A","vowel"],
    "ඈ": ["Aa","vowel"],
    "ඉ": ["i","vowel"],
    "ඊ": ["ii","vowel"],
    "උ": ["u","vowel"],
    "ඌ": ["uu","vowel"],
    "ඍ": ["Ri","vowel"],
    "එ": ["e","vowel"],
    "ඒ": ["ee","vowel"],
    "ඓ": ["ai","vowel"],
    "ඔ": ["o","vowel"],
    "ඕ": ["oo","vowel"],
    "ඖ": ["au","vowel"],
    "ං" : ["N","vowel"],
    "ඃ" : ["H","vowel"],
    "ක": ["ka", "consonant"],
    "ඛ": ["Ka", "consonant"],
    "ග": ["ga", "consonant"],
    "ඝ": ["Ga", "consonant"],
    "ඞ": ["Nga", "consonant"],
    "ඟ": ["nga", "consonant"],
    "ච": ["cha", "consonant"],
    "ඡ": ["Ca", "consonant"],
    "ජ": ["ja", "consonant"],
    "ඣ": ["Ja", "consonant"],
    "ඤ": ["Nya", "consonant"],
    "ඥ": ["nya", "consonant"],
    "ඦ": ["nna", "consonant"],
    "ට": ["ta", "consonant"],
    "ඨ": ["Ta", "consonant"],
    "ඩ": ["da", "consonant"],
    "ඪ": ["Da", "consonant"],
    "ණ": ["Na", "consonant"],
    "ඬ": ["nda", "consonant"],
    "ත": ["tha", "consonant"],
    "ථ": ["Tha", "consonant"],
    "ද": ["da", "consonant"],
    "ධ": ["Dha", "consonant"],
    "න": ["na", "consonant"],
    "ඳ": ["Nda", "consonant"],
    "ප": ["pa", "consonant"],
    "ඵ": ["Pa", "consonant"],
    "බ": ["ba", "consonant"],
    "භ": ["Bha", "consonant"],
    "ම": ["ma", "consonant"],
    "ඹ": ["mba", "consonant"],
    "ය": ["ya", "consonant"],
    "ර": ["ra", "consonant"],
    "ල": ["la", "consonant"],
    "ව": ["va", "consonant"],
    "ශ": ["sha", "consonant"],
    "ෂ": ["Sha", "consonant"],
    "ස": ["sa", "consonant"],
    "හ": ["ha", "consonant"],
    "ළ": ["La", "consonant"],
    "ෆ": ["fa", "consonant"],
    "්": ["", "glyph"],
    "ා": ["aa", "glyph"],
    "ැ": ["A", "glyph"],
    "ෑ": ["Aa", "glyph"],
    "ි": ["i", "glyph"],
    "ී": ["ii", "glyph"],
    "ු": ["u", "glyph"],
    "ූ": ["uu", "glyph"],
    "ෙ": ["e", "glyph"],
    "ේ": ["ee", "glyph"],
    "ෛ": ["ai", "glyph"],
    "ො": ["o", "glyph"],
    "ෝ": ["oo", "glyph"],
    "ෞ": ["au", "glyph"],
    "ෲ": ["Ri", "glyph"],
    "ෘ": ["Ru", "glyph"],
    "ෟ": ["Ru", "glyph"],
    "ෳ": ["Ruu", "glyph"],
    "෴": ["Ruu", "glyph"],
    "්‍ය": ["ya", "glyph"],
    "්‍ර": ["ra", "glyph"],
    "්‍ය": ["ya", "glyph"],
}


if __name__ == "__main__":
    sinhala_text = "නෝනා නාන්න නන් නාන්න නෝනා \n" + \
        "නෝනා නොනා නොනා නේන්න නෝනා \n"+\
        "නෑනා නානු නාන්නේ නා නා නානු නාන්නේ නෑනා නොනා නේන්නේ"
    print(sinhala_to_singlish(sinhala_text))

    print("==================================")

    sinhala_text = '''මා හා, 1 සෙනෙහස පාලා
    ඇය මා හා සිත බැඳුනා....
    සැනසීමක් සිතට දැනීලා
    පිරුනා සේ මට දැනුනා....'''
    print(sinhala_to_singlish(sinhala_text))

    print("==================================")

    sinahala_text = '''
    අම්මා සඳකි මම ඒ ලොව හිරුය රිදී
    ඒ ඉර හදෙන් නුඹෙ ලෝකය එළිය වුනි
    රැකුමට පුතුන් දිවියේ දුක් ගැහැට විදී
    පිය සෙනෙහසට කව් ගී ලියැවුනා මදී'''

    print(sinhala_to_singlish(sinahala_text))
    