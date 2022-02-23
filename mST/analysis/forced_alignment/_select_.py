import os
import random

with open('train-mismatch-v4.html', 'r') as r:
    html = [line for line in r.readlines() if line.strip() != '']

n_audio = (len(html) - 7) // 5
order = list(range(n_audio))
random.shuffle(order)

n_sample = 200

output = html[:6]
for o in order[:n_sample]:
    output.extend(html[6 + o * 5 : 6 + (o + 1) * 5])
output.append(html[-1])

with open('train-mismatch-sample-v4.html', 'w') as w:
    w.write(''.join(output))

os.makedirs('resources/train-mismatch-sample-v4/', exist_ok=True)

for o in order[:n_sample]:
    os.system('cp resources/train-mismatch-v4/audio_{}.wav resources/train-mismatch-sample-v4/audio_{}.wav'.format(o, o))
