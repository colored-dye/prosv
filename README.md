# Prompt-only Steering Vector

This repository provides the proof-of-concept code for *prompy-only steering vector (ProSV)*.

## Requirements

We recommend using `uv`.

Setup environment from scratch:

```
uv venv
source .venv/bin/activate
uv pip install -e .
```

... or use lockfile:

```
uv sync
```


## Generate with SV interventions

We have provided a **Jupyter noteboook** in `experiments/concept/concept.ipynb` for PrOSV with `p2+s2` interventions, i.e. two prompt prefix tokens and two prompt suffix tokens, as well as FSSV, along with their trained **checkpoints**.

- Concept: references to specific dates and publication information
- Model: `google/gemma-2-2b-it`
- Layer: 10
- Intervention location: `p2+s2`

Example case:

- Instruction: `"How can I make a cake?"`

| Original response                                            | PrOSV response                                                                                                                                                                                                                                     | FSSV response                                                                                                                                                                                                      |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Let's bake a cake!  Here's a basic guide to get you started. **1. Choose Your Recipe** * **Beginner:**  Look for recipes labeled "easy" or "simple" for cakes like vanilla, chocolate, or pound cake.  * **Intermediate:**  Try recipes with more complex flavors or decorations, like red velvet, lemon poppyseed, or bundt cakes.  * **Advanced:**  Explore recipes with unique ingredients or techniques, like layered cakes, cheesecakes, or sponge cakes.  **2. Gather Your Ingredients and Equipment** | A cake recipe, published in the "Joy of Cooking" cookbook in 1931, provides a classic base for baking. The book includes detailed instructions for preparing a vanilla cake, with specific steps outlined for mixing ingredients and baking times. Additionally, a 2015 article in "The New York Times" highlights the importance of using fresh ingredients and proper oven temperatures for optimal results. These resources offer valuable insights for creating a delicious cake, while a 2020 study published in "Food Science and Technology" emphasizes the impact of different baking techniques on texture and flavor. Therefore, incorporating these sources into your baking process can enhance your cake-making experience. \<eos\> | Baking a cake is a classic culinary practice that has been documented in cookbooks for centuries. A comprehensive guide to cake making can be found in "The Joy of Baking" by Irma S. Rombauer, published in 2021. Additionally, "Cake Decorating: A Comprehensive Guide" by Lisa Freeman, released in 2020, provides detailed instructions on frosting techniques and cake presentation. These resources offer insights into the evolution of cake baking, including the use of ingredients and the impact of temperature on the final product. Furthermore, "Baking Basics: A Beginner's Guide" by Betty Crocker, published in 2018, outlines fundamental steps for successful cake preparation. By consulting these publications, you can access a wealth of information on cake making, ensuring a delightful baking experience. |

