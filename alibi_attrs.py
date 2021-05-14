import numpy as np
from alibi.explainers import IntegratedGradients as alibi_ig
import matplotlib as mpl
from collections import Counter

def explain(tokenizer, essay, ig):
    attrs = get_attrs_alibi(tokenizer, essay, ig)[0]
    words,count = sequence_to_text(tokenizer, essay[0])
    assert len(words[count:]) == len(attrs[count:])
    html = visualize_token_attrs(words[count:], attrs[count:])
    return attrs, words, count, html

def get_attrs_alibi(tokenizer, essay, ig):
  baseline = np.zeros(essay.shape)
  baseline[0][0] = tokenizer.word_index['a']
  explanation = ig.explain(essay, baselines=baseline) 
  attrs = explanation.attributions
  attrs = attrs[0].sum(axis=2)
  return attrs
  
def sequence_to_text(tokenizer, list_of_indices):
    count = 0
    words = [tokenizer.index_word.get(ind) for ind in list_of_indices]
    for x in words:
      if x == None:
        count+=1
    return (words, count)
    
def visualize_token_attrs(tokens, attrs):
    cmap='PiYG'
    cmap_bound = np.abs(attrs).max()
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = mpl.cm.get_cmap(cmap)

    len_tokens = 0
    html_text = "<html><body><style type=\"text/css\"> p { display: inline-block;  width: 183pt;}</style> <p> "
    for i, tok in enumerate(tokens):
        if tok is not None:
          color = mpl.colors.rgb2hex(cmap(norm(attrs[i])))
          html_text += " <mark style='background-color:{}'>{}</mark>".format(color, tok)
    html_text+=" </p></body></html>"
    return (html_text)
