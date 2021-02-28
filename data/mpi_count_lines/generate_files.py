import random
num_files = 10
for file_ind in range(num_files):
    with open(f'file{file_ind}.txt', 'w') as f:
        chrs = 'qwertyuiopasdfghjklzxcvbnm0123456789 !?'
        lines = random.randint(30, 10000)
        for line in range(lines):
            num_letters = random.randint(10, 300)
            line_words = ''.join(random.choice(chrs) for _ in range(num_letters))
            line_words += '\n'
            f.write(line_words)
