# diffusion_ir



## How to run experiments on new datasets

1. Create json files of quries and corpus
- python create_ffhq_json.py 
2. Run genIR pipeline. (Multi rounds caption refinement)
- genIR_CaptionRefinment_TextOnly.py: Image -> Text
- genIR_CaptionImageRefinement.py: Text -> Fake Image
3. Eval 
- python ChatIR/eval_textonly_w_incompletete.py: text only eval. Supported incomplete queries
- python ChatIR/eval_img.py: Fake image to real image eval. Supported incomplete queries