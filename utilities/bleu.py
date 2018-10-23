from argparse import ArgumentParser
from nltk.translate.bleu_score import modified_precision
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

def calculate_bleu_from(ref_file, hyp_file):
    references = _read(ref_file)
    hypotheses = _read(hyp_file)
    assert len(references) == len(hypotheses)
    test_bleu_corpus(references, hypotheses)
    test_bleu_normal(references, hypotheses)
    
def get_sentence_bleu(reference, hypothesis):
    try:
        rew = sentence_bleu([reference.split()],hypothesis.split(), smoothing_function=SmoothingFunction().method4)    
    except: # fix for weird nltk bug
        rew = 0
    return rew

def get_corpus_bleu(references, hypotheses):
    return corpus_bleu(references, hypotheses)
    
def bleu(ref, hyp):
    return  float(modified_precision([ref.split()], hyp.split(), n=1))  

def test_bleu_normal(refs, hyps):
    res = [float(sentence_bleu([ref.split()], hyp.split())) for ref, hyp in zip(refs, hyps)] 
    print('normal {}'.format(sum(res)/len(refs)))

def test_bleu_corpus(references, hypotheses):
    ref = map(lambda x: [x.split()], references) 
    hyp = map(lambda x: x.split(), hypotheses) 
    print('corpus: {}'.format(corpus_bleu(ref, hyp)))

def _read(fl): 
    with open(fl, 'r') as f:
        return f.readlines()

def main():
    args = ArgumentParser()
    args.add_argument('hypotheses', type=str, help='The file containing output')
    args.add_argument('targets', type=str, help='The file containing the gold standards')
    parser = args.parse_args()
    calculate_bleu_from(parser.hypotheses, parser.targets)


if __name__ == '__main__':
    main()

