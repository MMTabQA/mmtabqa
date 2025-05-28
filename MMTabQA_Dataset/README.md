### Dataset Format
```
.
├── HybridQA
│   ├── explicit_ans_mention.jsonl => Contains the questions of Explicit Answer Mention type.
│   ├── explicit_questions.jsonl => Contains the questions of Explicit type.
│   ├── image_id_to_original_string.json => Contains a mapping of the image_id to the original string of the entity replaced. Can be used for verification/analysis.
│   ├── image_id_to_qid.json => Contains a mapping of the image_id to the Wikidata ID of the entity replaced. Can be used for verification/analysis.
│   ├── image_id_to_wikipedia_link.json => Contains a mapping of the image_id to the Wikipedia link of the entity replaced. Can be used for verification/analysis.
│   ├── implicit_questions.jsonl => Contains the questions of the implicit type.
│   ├── mm_passages.json => Contains the Multimodal Passages corresponding to entities mentioned in the tables for HybridQA. These are to be used for the entities which have been replaced by images in the HybridQA tables. **Only present in HybridQA**.
│   ├── tables.jsonl => Contains all the tables of the dataset.
│   ├── text_passages.zip => Contains the Multimodal Passages corresponding to entities mentioned in the tables for HybridQA. These are to be used for the entities which have been replaced by images in the HybridQA tables. **Only present in HybridQA**.
│   └── visual_questions.jsonl => Contains the questions of Visual type.
├── image_id_to_image_path.json => Contains a mapping of image_id used in the tables to Image path. The images are available [here](https://mega.nz/file/TB8FUb7D#D7REnXhbJbd8mR6KkcJbdOkIGHCuY3mJkxQjR_39-2o).
```