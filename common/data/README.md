# common/data

Download / clean / convert scripts, and custom data processors.

NeMo-RL 0.7.0:

- Builtin datasets: `data.train.dataset_name=...` (`squad`, `OpenMathInstruct-2`, …).
- Local files:
  - GRPO `ResponseDataset`: `data.train.data_path`, `data.train.input_key`, `data.default.dataset_name=ResponseDataset`.
  - SFT OpenAI messages: `data.train_data_path`, `data.chat_key=messages`.
- Custom processor: implement one and point `data.default.processor` at it (`math_hf_data_processor` / `sft_processor` in upstream).

Scripts are in git. Raw / large outputs are not. `sf dataset prepare <name>` runs `prepare_<name>.py` and writes `datasets/<name>/`.
