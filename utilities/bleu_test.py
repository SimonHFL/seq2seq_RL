import bleu


references = ["this is a sentence with a lot of words",
				"another sentence with many other words"]

candidates = ["this is a sentence with some words","another sentence with many other words"]

references = [r.split() for r in references]
candidates = [c.split() for c in candidates]

print(bleu.get_corpus_bleu(references, candidates))