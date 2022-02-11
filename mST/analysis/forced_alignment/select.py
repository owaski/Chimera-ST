import os
import random

with open('train-mismatch.html', 'r') as r:
    html = [line for line in r.readlines() if line.strip() != '']

n_audio = (len(html) - 7) // 5
order = list(range(n_audio))
random.shuffle(order)

n_sample = 100

output = html[:6]
for o in order[:n_sample]:
    output.extend(html[6 + o * 5 : 6 + (o + 1) * 5])
output.append(html[-1])

with open('trian-mismatch-sample.html', 'w') as w:
    w.write(''.join(output))

os.makedirs('resources/train-mismatch-sample', exist_ok=True)

for o in order[:n_sample]:
    os.system('cp resources/train-mismatch/audio_{}.wav resources/train-mismatch-sample/audio_{}.wav'.format(o, o))
