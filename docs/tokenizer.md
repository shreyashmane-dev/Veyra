# VeyraTokenizer: Byte-Level BPE Tokenizer

VEYRA utilizes its own Byte-Level Byte-Pair Encoding (BPE) tokenizer trained directly on the project corpus.

## Key Features

1. **Zero Out-of-Vocabulary (OOV) Guarantee**:
   - The base vocabulary registers all 256 possible UTF-8 byte values `<0x00>` to `<0xFF>`.
   - Any arbitrary binary sequence or unseen foreign Unicode script can be encoded and decoded without failure.
2. **Special Semantic Tokens**:
   - `<PAD>`: ID 0
   - `<UNK>`: ID 1
   - `<BOS>`: ID 2
   - `<EOS>`: ID 3
   - `<USER>`: ID 4
   - `<ASSISTANT>`: ID 5
   - `<SYSTEM>`: ID 6
   - `<TOOL>`: ID 7
   - `<THOUGHT>`: ID 8
3. **CLI Utilities**:
   ```bash
   # Train tokenizer
   python -m veyra.cli.main tokenizer train --corpus data/raw/bootstrap_corpus.txt --vocab-size 1024 --output data/tokenizer

   # Inspect tokenizer
   python -m veyra.cli.main tokenizer inspect --path data/tokenizer

   # Encode text
   python -m veyra.cli.main tokenizer encode --path data/tokenizer --text "Hello VEYRA"

   # Decode token IDs
   python -m veyra.cli.main tokenizer decode --path data/tokenizer --ids "2, 81, 295, 3"
   ```
