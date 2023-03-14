import random
import json

with open('styles.json', 'r', encoding='utf-8') as f_in:
    styles = json.load(f_in)

with open('styles_count.json', 'r', encoding='utf-8') as f_in:
    styles_count = json.load(f_in)


proba = {}
for k in styles:
    proba[k] = []
    for n, item in enumerate(styles[k]):
        proba_n = (styles_count[k][n] + 1) / (sum(styles_count[k]) + len(styles_count[k]))
        proba[k].append(proba_n)

[print(i, sum(proba[i])) for i in proba]

result_style = {}
for feature in styles:
    p_acc = proba[feature]
    acc = styles[feature]
    result_style[feature] = random.choices(acc, p_acc)[0]

if result_style['прическа'] == 'нет волос':
    result_style.pop('цвет волос')

print(result_style)
