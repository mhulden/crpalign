# crpalign
One-to-one-aligner through MCMC

# Compile C code to get Python bindings
Something like:
```gcc -O3 -Wall -Wextra -shared align.c -o libalign.so```

# Usage
See ```example.py``` which aligns English words and their IPA pronunciations, producing pairs like:

```
j _ u θ _
y o u t h

j ɑ _ _ t
y a c h t

w _ ɝ ʃ _ ə p
w o r s h i p

ɑ n s l ɔ _ _ _ t
o n s l a u g h t
```
 
