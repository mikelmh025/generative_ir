
# ChatIR - MSCOCO Images
# Caption refinement text only
python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-12b-it --output_dir=results_genir/self_dialogue_caption_output_gemma_3_12b_it --gpu_id=0

python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-4b-it --output_dir=results_genir/self_dialogue_caption_output_gemma_3_4b_it --gpu_id=1

python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-4b-it --output_dir=results_genir/self_dialogue_caption_output_gemma_3_4b_it --gpu_id=1

####################################################################################################

# FFHQ Run - Text Only
python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-12b-it --input_dir='/data/skin_gen_data/ffhq/images' --output_dir=results_genir/self_dialogue_caption_ffhq-12b-q2k --gpu_id=1 --max_images=1000 --queries_path='ffhq_data/ffhq_queries_2000.json' --refinement_rounds=10

python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-4b-it --input_dir='/data/skin_gen_data/ffhq/images' --output_dir=results_genir/self_dialogue_caption_ffhq-4b-q2k --gpu_id=0 --max_images=1000 --queries_path='ffhq_data/ffhq_queries_2000.json' --refinement_rounds=10

# FFHQ Run Ours Pipeline
# --image_subdirs?
language_model='google/gemma-3-12b-it'
output_dir='results_genir/caption_refinement_output_infinity-12b-FFHQ-q2k'
# language_model='google/gemma-3-4b-it'
# output_dir='results_genir/caption_refinement_output_infinity-4b-FFHQ-q2k'
python genIR_CaptionImageRefinement_wIF.py \
    --input_dir='/data/skin_gen_data/ffhq/images' \
    --queries_path='ffhq_data/ffhq_queries_2000.json' \
    --caption_model_id=$language_model \
    --comparison_model_id=$language_model \
    --output_dir=$output_dir \
    --max_images=3000 


# FFHQ Eval - Text Only
# external_text_root=results_genir/self_dialogue_caption_ffhq-4b-q2k/captions # 4b
external_text_root=results_genir/self_dialogue_caption_ffhq-12b-q2k/captions # 12b

# external_text_root=results_genir/self_dialogue_caption_ffhq-4b/captions # 4b initial try
# external_text_root=results_genir/self_dialogue_caption_ffhq-12b/captions # 12b initial try

python ChatIR/eval_textonly_w_incompletete.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small_ffhq.pth' \
    --queries_path='ffhq_data/ffhq_queries_2000.json' \
    --corpus_path='ffhq_data/ffhq_corpus_all.json' \
    --img_root='/data/skin_gen_data/ffhq/images' \
    --corpus_bs=500 \
    --num_workers=64 \
    --external_text_root=$external_text_root \
    --hits_at=10


    # --queries_path='ffhq_data/self_dialogue_caption_ffhq-12b-queries.json' \
    # --queries_path='ffhq_data/ffhq_queries_2000.json' \


# FFHQ Eval - Fake Images
fake_images_dir='results_genir/caption_refinement_output_infinity-4b-FFHQ-q2k/generated_images/'
python ChatIR/eval_img.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small_ffhq.pth' \
    --queries_path='ffhq_data/ffhq_queries_2000.json' \
    --corpus_path='ffhq_data/ffhq_corpus_all.json' \
    --img_root='/data/skin_gen_data/ffhq/images' \
    --corpus_bs=500 \
    --num_workers=64 \
    --fake_images_dir=$fake_images_dir \
    --hits_at=10


####################################################################################################

# ClothingADC  dataset
python create_clothingADC_json.py \
    --clothing-dir='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --output-dir='clothingadc_data' \
    --corpus-size=60000 \
    --query-size=2000 



# ClothingADC Run - Text Only
# For 12B model
python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-12b-it --input_dir='/data/cloth1m/v4/cloth1m_data_v4/images' --output_dir=results_genir/self_dialogue_caption_clothingADC-12b-q2k --gpu_id=1 --max_images=1000 --queries_path='clothingadc_data/clothingadc_queries_2000.json' --refinement_rounds=10

# For 4B model [Running]
python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-4b-it --input_dir='/data/cloth1m/v4/cloth1m_data_v4/images' --output_dir=results_genir/self_dialogue_caption_clothingADC-4b-q2k --gpu_id=1 --max_images=1000 --queries_path='clothingadc_data/clothingadc_queries_2000.json' --refinement_rounds=10


# Full Pipeline with Image Generation
# For 12B model
language_model='google/gemma-3-12b-it'
output_dir='results_genir/caption_refinement_output_infinity-12b-clothingADC-q2k'
python genIR_CaptionImageRefinement_wIF.py \
    --input_dir='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --queries_path='clothingadc_data/clothingadc_queries_2000.json' \
    --caption_model_id=$language_model \
    --comparison_model_id=$language_model \
    --output_dir=$output_dir \
    --max_images=3000

# For 4B model
language_model='google/gemma-3-4b-it'
output_dir='results_genir/caption_refinement_output_infinity-4b-clothingADC-q2k'
python genIR_CaptionImageRefinement_wIF.py \
    --input_dir='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --queries_path='clothingadc_data/clothingadc_queries_2000.json' \
    --caption_model_id=$language_model \
    --comparison_model_id=$language_model \
    --output_dir=$output_dir \
    --max_images=3000


# ClothingADC Eval - Text Only
# For 12B model
external_text_root=results_genir/self_dialogue_caption_clothingADC-12b-q2k/captions
python ChatIR/eval_textonly_w_incompletete.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small_clothingADC.pth' \
    --queries_path='clothingadc_data/clothingadc_queries_2000.json' \
    --corpus_path='clothingadc_data/clothingadc_corpus_60000.json' \
    --img_root='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --corpus_bs=500 \
    --num_workers=64 \
    --external_text_root=$external_text_root \
    --hits_at=10

# For 4B model
external_text_root=results_genir/self_dialogue_caption_clothingADC-4b-q2k/captions
python ChatIR/eval_textonly_w_incompletete.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small_clothingADC_all.pth' \
    --queries_path='clothingadc_data/clothingadc_queries_2000.json' \
    --corpus_path='clothingadc_data/clothingadc_corpus_60000.json' \
    --img_root='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --corpus_bs=500 \
    --num_workers=64 \
    --external_text_root=$external_text_root \
    --hits_at=10

# ClothingADC Eval - Fake Images
# For 4B model
fake_images_dir='results_genir/caption_refinement_output_infinity-4b-clothingADC-q2k/generated_images/'
python ChatIR/eval_img.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small_clothingADC_all.pth' \
    --queries_path='clothingadc_data/clothingadc_queries_2000.json' \
    --corpus_path='clothingadc_data/clothingadc_corpus_all.json' \
    --img_root='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --corpus_bs=500 \
    --num_workers=64 \
    --fake_images_dir=$fake_images_dir \
    --hits_at=10

# For 12B model
fake_images_dir='results_genir/caption_refinement_output_infinity-12b-clothingADC-q2k/generated_images/'
python ChatIR/eval_img.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small_clothingADC.pth' \
    --queries_path='clothingadc_data/clothingadc_queries_2000.json' \
    --corpus_path='clothingadc_data/clothingadc_corpus_60000.json' \
    --img_root='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --corpus_bs=500 \
    --num_workers=64 \
    --fake_images_dir=$fake_images_dir \
    --hits_at=10