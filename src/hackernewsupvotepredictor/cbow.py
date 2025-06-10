
import numpy as np
import matplotlib.pyplot as plt
'''
import urllib.request
url = 'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8'
response = urllib.request.urlopen(url)
data = response.read()      # a bytes object
text = data.decode('utf-8')
'''

with open("data/text8", "r") as f:
  raw_data = f.read()

test = raw_data[:10000]


def tokenizer(text):
    # Stemming
    result = text.lower().replace("ing ", " ").replace("'s ", " ").replace("es ", "e ").replace("ied ", "y ")
    # Replace contents
    result = result.replace(", ", " <COMMA> ").replace(". ", " <FULLSTOP> ").replace("'", " <APOSTROPHE> ").replace("!", " <EXCLAMATION>").replace("? ", " <QUESTION> ").replace(": ", " <ELIPSES> ")
    result = result.replace(" aaaaaacceglllnorst ", " <UNKNOWN> ").replace(" aaaaaaccegllnorrst ", " <UNKNOWN> ").replace("  aaaaaah ", " <UNKNOWN> ").replace(" zzurf  ", " <UNKNOWN> ").replace(" zzum  ", " <UNKNOWN> ").replace(" ...  ", " ")
    return np.array(result.split(" "))

updtext = tokenizer(test)
unique, counts = np.unique(updtext, return_counts=True)
plt.hist(counts)
plt.show()







