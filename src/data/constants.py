'''
keep project-wide constants in one place so they are not duplicated throughout the code

Contains:

label mappings
class names
maybe supported image extensions
'''

LABEL_TO_INDEX = {
    "Dominant_Follicle": 0,
    "Normal": 1,
    "PCO": 2,
}

INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}