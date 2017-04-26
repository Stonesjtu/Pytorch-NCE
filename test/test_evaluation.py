from main import evaluate, corpus


def test_evaluate():
    assert corpus.test
    assert evaluate(corpus.test)
