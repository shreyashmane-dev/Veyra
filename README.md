# VEYRA

VEYRA is a personal AI project designed as a small, self-built alternative to a general AI assistant.

## 0.1 goals
- Runs completely in the terminal
- Keeps conversational memory in SQLite
- Learns user-defined facts and rules with `teach`
- Has a modular cognitive architecture
- Includes a model interface so a self-trained VEYRA language model can replace the starter engine later
- Leaves room for voice, vision, tools, planning, and web access

## Run

```bash
python -m veyra
```

From the project root.

## Commands

```text
/help       show commands
/memory     show remembered facts
/teach      teach VEYRA a fact or rule
/status     show cognitive state
/clear      clear current conversation
/quit       exit
```

Normal text is handled by the current starter language engine.

## Roadmap

1. Terminal + memory + teaching
2. Own tokenizer
3. Tiny Transformer language model (VEYRA-LM)
4. Dataset pipeline and training on permitted datasets
5. Conversation/instruction tuning
6. Reasoning and planning
7. Tool use and safe computer actions
8. Speech-to-text and text-to-speech
9. Vision/document understanding
10. Larger multimodal personal assistant
