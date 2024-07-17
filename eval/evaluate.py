import argparse
import json
import os
import subprocess
import sys
from tempfile import NamedTemporaryFile

import conll18_ud_eval as ud_eval

_CONLL_EVAL_SCRIPT = os.path.join(os.path.abspath(os.path.dirname(__file__)), "eval.pl")


def evaluate_conll(gold_file, system_file, verbose=False):
    command = ["/usr/bin/perl", _CONLL_EVAL_SCRIPT, "-g", gold_file, "-s", system_file]
    if not verbose:
        command.append("-q")
    option = {}
    p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **option)
    output = (p.stdout if p.returncode == 0 else p.stderr).decode("utf-8")
    if p.returncode != 0:
        error = p.stderr.decode("utf-8")
        raise RuntimeError("code={!r}, message={!r}".format(p.returncode, error))
    output = p.stdout.decode("utf-8")
    scores = [float(line.rsplit(" ", 2)[-2]) for line in output.split("\n", 2)[:2]]
    return {"LAS": scores[0], "UAS": scores[1], "raw": output}


def evaluate_conllu(gold_file, system_file, verbose=False):
    gold_ud = ud_eval.load_conllu_file(gold_file)
    system_ud = ud_eval.load_conllu_file(system_file)
    evaluation = ud_eval.evaluate(gold_ud, system_ud)
    output = ud_eval.build_evaluation_table(evaluation, verbose, counts=False)
    return {"LAS": evaluation["LAS"].f1, "UAS": evaluation["UAS"].f1, "raw": output}


def dump_conll(parses, writer=sys.stdout):
    attrs = ["id", "form", "lemma", "cpostag", "postag", "feats", "head", "deprel"]
    if len(attrs) < 10:
        attrs += [None] * (10 - len(attrs))

    for example in parses:
        for i in range(len(example["id"])):
            values = (str(example[k][i]) if k in example else "_" for k in attrs)
            writer.write("\t".join(values) + "\n")
        writer.write("\n")
    writer.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file")
    parser.add_argument("pred_file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with open(args.pred_file) as f:
        predictions = [json.loads(line) for line in f]

    with NamedTemporaryFile(mode="w") as f:
        dump_conll(predictions, f)
        evaluate = evaluate_conllu if args.gold_file.endswith(".conllu") else evaluate_conll
        result = evaluate(args.gold_file, f.name, verbose=args.verbose)

    print(result["raw"] if args.verbose else f"LAS={result['LAS']}, UAS={result['UAS']}")


if __name__ == "__main__":
    main()
