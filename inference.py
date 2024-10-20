import sys,librosa,torch
import numpy as np 
from g2p_en import G2p
sys.path.append('/Users/samarasimhareddygujjula/Desktop/charsiu/src')

from Charsiu import charsiu_forced_aligner,charsiu_predictive_aligner
import nltk
# nltk.download('averaged_perceptron_tagger_eng')


# initialize model
charsiu = charsiu_predictive_aligner(aligner='charsiu/en_w2v2_fc_10ms')


audio,sr=librosa.load('/Users/samarasimhareddygujjula/Desktop/charsiu/train_bengalimale_01900.wav',sr=16000)

# alignment = charsiu.align(audio=audio,
#                           text='The name of the story is A Contest Of With .')
alignment= charsiu.align(audio=audio)

predicted_phonemes = [phoneme for _, _, phoneme in alignment if phoneme != '[SIL]']

print(alignment,"align 0")

with open('ipa_text_bengali.txt','w') as f:
    for start,end,token in alignment:
        f.write(str(start) + " " + str(end) + " " + token + '\n')


