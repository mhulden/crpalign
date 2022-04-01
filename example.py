import align

lines = [l.strip().split() for l in open("englishpron.txt")]
pairs = [(spelling, pron) for pron, spelling in lines]

ap = align.Aligner(lines, '_', iterations = 50, mode = 'crp')
for input, output in ap.alignedpairs:
    print(' '.join(input) + '\n' + ' '.join(output) + '\n')

