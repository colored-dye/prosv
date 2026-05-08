# Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions

<div align="center">

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.05983)  [![Github](https://img.shields.io/badge/PrOSV_Code-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/colored-dye/prosv)  [![Hugging Face Data](https://img.shields.io/badge/Data-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/colored-dye/concept500-contrastive)  [![Hugging Face Model](https://img.shields.io/badge/SV_Checkpoints-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/colored-dye/axbench-steering-vector)

</div>


Representation steering with support for KV cache :zap:.

Features :sparkles: :

* Steering vectors (SVs) with two intervention location strategies:
  * Prompt-only: PrOSV.
  * Full-sequence: FSSV.
* LLM-judge evaluation: AxBench protocol (overall steering score, i.e. the harmonic mean of concept/instruct/fluency scores).


## Acknowledgements

Many thanks to these projects; the community is better for your valuable contributions :pray: :

* [baukit](https://github.com/davidbau/baukit)
* [pyvene](https://github.com/stanfordnlp/pyvene)
* [pyreft](https://github.com/stanfordnlp/pyreft)
* [axbench](https://github.com/stanfordnlp/axbench)


## Citation

If you find our work useful, please cite as:

```bibtex
@inproceedings{bao2026towards,
  title = {Towards Steering without Sacrifice: Principled Training of Steering Vectors for Prompt-only Interventions},
  author = {Bao, Yuntai and Li, Qinfeng and Yu, Xinyan and Zhang, Xuhong and Su, Ge and Zhang, Wenqi and Yan, Liu and Weng, Haiqin and Yin, Jianwei},
  booktitle = {Forty-third International Conference on Machine Learning},
  year = {2026},
  url = {https://openreview.net/forum?id=AaT3liS5PE},
}
```
