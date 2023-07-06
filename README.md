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

@misc{chakraborty2023zeroshot,
      title={Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts}, 
      author={Mohna Chakraborty and Adithya Kulkarni and Qi Li},
      year={2023},
      eprint={2305.15689},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

<b>We will add ACL citation soon.</b>
</html>