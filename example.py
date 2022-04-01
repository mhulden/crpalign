import align

lines = [l.strip().split() for l in open("englishpron.txt")]

ap = align.Aligner(lines, '_', iterations = 50, mode = 'crp')
for input, output in ap.alignedpairs:
    print(' '.join(input) + '\n' + ' '.join(output) + '\n')

