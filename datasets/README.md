# datasets/

Metadata and small files only. Raw / processed blobs stay on disk or object storage (gitignored).

```
datasets/<name>/
├── README.md      source, version, license, fields, prepare command
├── train.jsonl    only if small; otherwise gitignore + how to get it
└── raw/
```

`sf dataset prepare <name>` runs `common/data/prepare_<name>.py`.
