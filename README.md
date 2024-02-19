<html>
<h1>Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts (ACL 2023)</h1>

<h2>Folder Structures</h2>
<li><b>code:</b> This folder contains the code for the paper.</li>
<li><b>dataset:</b> Download and keep the datasets into this folder.</li>
<li><b>files:</b> Generated files will be saved in this folder.</li>

<h2>Datasets</h2>
Get the following datasets from their respective papers and keep them in the <b>dataset</b> folder.
<li><b>SST-2:</b> The dataset is provided in the paper "Recursive deep models for
semantic compositionality over a sentiment treebank".</li>
<li><b>CR:</b> The dataset is provided in the paper "Mining and
summarizing customer reviews".</li>
<li><b>MR:</b> The dataset is provided in the paper "Thumbs up? sentiment classification
using machine learning techniques".</li>

<h2>Execution Sequence</h2>
<li><b>generate_templates.py:</b> Run this python script to automatically generate templates.</li>
<li><b>template_scores.py</b> Run this python script to obtain scores for the automatically generated templates. Use these scores to rank the templates.</li>
<li><b>evaluate_templates.py</b> Run this python script to evaluate the automatically generated templates with respect to ground truth.</li>

<h2>Cite</h2>
Please cite our paper:
```@inproceedings{chakraborty-etal-2023-zero,
    title = "Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts",
    author = "Chakraborty, Mohna  and Kulkarni, Adithya  and Li, Qi",
    editor = "Rogers, Anna  and  Boyd-Graber, Jordan  and Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.313",
    doi = "10.18653/v1/2023.acl-long.313",
    pages = "5698--5711",
    abstract = "Recent studies have demonstrated that natural-language prompts can help to leverage the knowledge learned by pre-trained language models for the binary sentence-level sentiment classification task. Specifically, these methods utilize few-shot learning settings to fine-tune the sentiment classification model using manual or automatically generated prompts. However, the performance of these methods is sensitive to the perturbations of the utilized prompts. Furthermore, these methods depend on a few labeled instances for automatic prompt generation and prompt ranking. This study aims to find high-quality prompts for the given task in a zero-shot setting. Given a base prompt, our proposed approach automatically generates multiple prompts similar to the base prompt employing positional, reasoning, and paraphrasing techniques and then ranks the prompts using a novel metric. We empirically demonstrate that the top-ranked prompts are high-quality and significantly outperform the base prompt and the prompts generated using few-shot learning for the binary sentence-level sentiment classification task.",
}```
</html>
