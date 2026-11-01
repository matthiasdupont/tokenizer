from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=5000,
    min_frequency=2,
    special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"]
)

tokenizer.train(["ai_corpus_cleaned.txt"], trainer)

# Save the trained tokenizer to a file
tokenizer.save("ai_tokenizer.json")
