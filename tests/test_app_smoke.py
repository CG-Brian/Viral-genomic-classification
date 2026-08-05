def test_app_exposes_demo():
    import app

    assert app.demo is not None
    assert callable(app.run_prediction)


def test_callback_returns_prediction_outputs():
    import app

    outputs = app.run_prediction("ACGT" * 38)
    assert len(outputs) == 6
    assert "Predicted read class" in outputs[0]
    assert "Model score" in outputs[1]
    assert outputs[3] is not None
    assert outputs[4] is not None


def test_callback_returns_validation_message():
    import app

    outputs = app.run_prediction("NOTASEQUENCE" * 2)
    assert outputs[0] == "No prediction"
    assert outputs[3] is None
    assert "Unsupported nucleotide" in outputs[5]
