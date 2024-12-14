def get_hf_model_tokenizer(model_name, checkpoint=None):
    assert model_name in ["hyenadna", "mambadna", "dnabert2"]
    if model_name == "hyenadna":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
    elif model_name == "mambadna":
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True)
    elif model_name == "dnabert2":
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )

    return model, tokenizer
