sudo du -h --max-depth=1 | sort -hr

sudo sync; echo 1 | sudo tee /proc/sys/vm/drop_caches
sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches


# ChatIR - MSCOCO Images
# Caption refinement text only
python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-12b-it --output_dir=results_genir/self_dialogue_caption_output_gemma_3_12b_it --gpu_id=0

python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-4b-it --output_dir=results_genir/self_dialogue_caption_output_gemma_3_4b_it --gpu_id=1

python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-4b-it --output_dir=results_genir/self_dialogue_caption_output_gemma_3_4b_it --gpu_id=1

python genIR_CaptionImageRefinement.py --caption_model_id=google/gemma-3-4b-it --comparison_model_id=google/gemma-3-4b-it --output_dir=results_genir/caption_refinement_output_infinity_4b

python genIR_CaptionRefinment_VIsualPredictionFeedBack.py --caption_model_id=google/gemma-3-12b-it --output_dir=results_genir/visual_prediction_feedback_output_12b

# external_text_root=results_genir/visual_prediction_feedback_output/captions
# external_text_root=results_genir/visual_prediction_feedback_output_12b/captions
# external_text_root=results_genir/caption_refinement_output_infinity/captions
# external_text_root=results_genir/caption_refinement_output_infinity_4b/captions
external_text_root=results_genir/self_dialogue_caption_output_gemma_3_12b_it/captions
python ChatIR/eval_textonly_w_incompletete.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small.pth' \
    --corpus_bs=500 \
    --num_workers=64 \
    --external_text_root=$external_text_root \
    --hits_at=10


# MSCOCO Eval - Fake Images
fake_images_dir='results_genir/caption_refinement_output_infinity_4b/generated_images/'
python ChatIR/eval_img.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small.pth' \
    --corpus_bs=500 \
    --num_workers=64 \
    --fake_images_dir=$fake_images_dir \
    --hits_at=10


####################################################################################################

# FFHQ Run - Text Only
python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-12b-it --input_dir='/data/skin_gen_data/ffhq/images' --output_dir=results_genir/self_dialogue_caption_ffhq-12b-q2k --gpu_id=1 --max_images=1000 --queries_path='ffhq_data/ffhq_queries_2000.json' --refinement_rounds=10

python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-4b-it --input_dir='/data/skin_gen_data/ffhq/images' --output_dir=results_genir/self_dialogue_caption_ffhq-4b-q2k --gpu_id=0 --max_images=1000 --queries_path='ffhq_data/ffhq_queries_2000.json' --refinement_rounds=10

# FFHQ Run - Visual Prediction Feedback
python genIR_CaptionRefinment_VIsualPredictionFeedBack.py --caption_model_id=google/gemma-3-4b-it --output_dir=results_genir/visual_prediction_feedback_output_4b_FFHQ --caption_gpu_id=0 --retrieval_gpu_id=0 --max_images=500 --input_dir='/data/skin_gen_data/ffhq/images' --max_images=500 --queries_path='ffhq_data/ffhq_queries_2000.json' --cache_corpus='temp/corpus_blip_small_ffhq.pth' --corpus_path='ffhq_data/ffhq_corpus_all.json' --corpus_bs=500 --num_workers=64

# FFHQ Run - Visual Prediction Feedback 12B
python genIR_CaptionRefinment_VIsualPredictionFeedBack.py --caption_model_id=google/gemma-3-12b-it --output_dir=results_genir/visual_prediction_feedback_output_12b_FFHQ --caption_gpu_id=1 --retrieval_gpu_id=1 --max_images=500 --input_dir='/data/skin_gen_data/ffhq/images' --max_images=500 --queries_path='ffhq_data/ffhq_queries_2000.json' --cache_corpus='temp/corpus_blip_small_ffhq.pth' --corpus_path='ffhq_data/ffhq_corpus_all.json' --corpus_bs=500 --num_workers=64

# FFHQ Run Ours Pipeline
# --image_subdirs?
language_model='google/gemma-3-12b-it'
output_dir='results_genir/caption_refinement_output_infinity-12b-FFHQ-q2k'
# language_model='google/gemma-3-4b-it'
# output_dir='results_genir/caption_refinement_output_infinity-4b-FFHQ-q2k'
python genIR_CaptionImageRefinement.py \
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

# external_text_root=results_genir/visual_prediction_feedback_output_4b_FFHQ/captions
external_text_root=results_genir/visual_prediction_feedback_output_12b_FFHQ/captions

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



# FFHQ Eval - Fake Images
# fake_images_dir='results_genir/caption_refinement_output_infinity-4b-FFHQ-q2k/generated_images/'
fake_images_dir='results_genir/caption_refinement_output_infinity-12b-FFHQ-q2k/generated_images/'
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
python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-12b-it --input_dir='/data/cloth1m/v4/cloth1m_data_v4/images' --output_dir=results_genir/self_dialogue_caption_clothingADC-12b-q2k --gpu_id=0 --max_images=500 --queries_path='clothingadc_data/clothingadc_queries_2000.json' --refinement_rounds=10

# For 4B model 
python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-4b-it --input_dir='/data/cloth1m/v4/cloth1m_data_v4/images' --output_dir=results_genir/self_dialogue_caption_clothingADC-4b-q2k --gpu_id=1 --max_images=1000 --queries_path='clothingadc_data/clothingadc_queries_2000.json' --refinement_rounds=10

# ClothingADC Run - Visual Prediction Feedback
python genIR_CaptionRefinment_VIsualPredictionFeedBack.py --caption_model_id=google/gemma-3-4b-it --output_dir=results_genir/visual_prediction_feedback_output_4b_clothingADC --caption_gpu_id=1 --retrieval_gpu_id=1 --max_images=500 --input_dir='/data/cloth1m/v4/cloth1m_data_v4/images' --max_images=500 --queries_path='clothingadc_data/clothingadc_queries_2000.json' --cache_corpus='temp/corpus_blip_small_clothingADC_all.pth' --corpus_path='clothingadc_data/clothingadc_corpus_all.json' --corpus_bs=500 --num_workers=64




# Full Pipeline with Image Generation
# For 12B model
language_model='google/gemma-3-12b-it'
output_dir='results_genir/caption_refinement_output_infinity-12b-clothingADC-q2k'
python genIR_CaptionImageRefinement.py \
    --input_dir='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --queries_path='clothingadc_data/clothingadc_queries_2000.json' \
    --caption_model_id=$language_model \
    --comparison_model_id=$language_model \
    --output_dir=$output_dir \
    --max_images=500

# For 4B model
language_model='google/gemma-3-4b-it'
output_dir='results_genir/caption_refinement_output_infinity-4b-clothingADC-q2k'
python genIR_CaptionImageRefinement.py \
    --input_dir='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --queries_path='clothingadc_data/clothingadc_queries_2000.json' \
    --caption_model_id=$language_model \
    --comparison_model_id=$language_model \
    --output_dir=$output_dir \
    --max_images=500


# ClothingADC Eval - Text Only
# For 12B model
external_text_root=results_genir/self_dialogue_caption_clothingADC-12b-q2k/captions
python ChatIR/eval_textonly_w_incompletete.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small_clothingADC_all.pth' \
    --queries_path='clothingadc_data/clothingadc_queries_2000.json' \
    --corpus_path='clothingadc_data/clothingadc_corpus_all.json' \
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
    --corpus_path='clothingadc_data/clothingadc_corpus_all.json' \
    --img_root='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --corpus_bs=500 \
    --num_workers=64 \
    --external_text_root=$external_text_root \
    --hits_at=10

# Visual Prediction Feedback
external_text_root=results_genir/visual_prediction_feedback_output_4b_clothingADC/captions
python ChatIR/eval_textonly_w_incompletete.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small_clothingADC_all.pth' \
    --queries_path='clothingadc_data/clothingadc_queries_2000.json' \
    --corpus_path='clothingadc_data/clothingadc_corpus_all.json' \
    --img_root='/data/cloth1m/v4/cloth1m_data_v4/images' \
    --corpus_bs=500 \
    --num_workers=64 \
    --external_text_root=$external_text_root \
    --hits_at=10

# ClothingADC Eval - Fake Images
# For 4B model
fake_images_dir='results_genir/caption_refinement_output_infinity-4b-clothingADC-q2k/generated_images/'
python ChatIR/eval_img.py \
    --cuda_device='0' \
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


#############################################
##### Flicker30k

# First create the JSON files needed for queries and corpus
python create_flickr30k_json.py --output-dir='flickr30k_data'

# Flickr30k Run - Text Only
python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-12b-it --input_dir='/data/flickr30k/flickr30k-images' --output_dir=results_genir/self_dialogue_caption_flickr30k-12b --gpu_id=0 --max_images=500 --queries_path='flickr30k_data/flickr30k_queries_2000.json' --refinement_rounds=10

# Running the 4B model
python genIR_CaptionRefinment_TextOnly.py --model_id=google/gemma-3-4b-it --input_dir='/data/flickr30k/flickr30k-images' --output_dir=results_genir/self_dialogue_caption_flickr30k-4b --gpu_id=0 --max_images=500 --queries_path='flickr30k_data/flickr30k_queries_2000.json' --refinement_rounds=10

# Flickr30k Run - Visual Prediction Feedback
python genIR_CaptionRefinment_VIsualPredictionFeedBack.py --caption_model_id=google/gemma-3-4b-it --output_dir=results_genir/visual_prediction_feedback_output_4b_flickr30k --caption_gpu_id=0 --retrieval_gpu_id=0 --max_images=500 --input_dir='/data/flickr30k/flickr30k-images' --max_images=500 --queries_path='flickr30k_data/flickr30k_queries_2000.json' --cache_corpus='temp/corpus_blip_small_flickr30k.pth' --corpus_path='flickr30k_data/flickr30k_corpus_all.json' --corpus_bs=500 --num_workers=64

# Flickr30k Run - Full Pipeline with Image Generation
# Option 1: 12B model
language_model='google/gemma-3-12b-it'
output_dir='results_genir/caption_refinement_output_infinity-12b-flickr30k'
python genIR_CaptionImageRefinement.py \
    --input_dir='/data/flickr30k/flickr30k-images' \
    --queries_path='flickr30k_data/flickr30k_queries_2000.json' \
    --caption_model_id=$language_model \
    --comparison_model_id=$language_model \
    --output_dir=$output_dir \
    --max_images=500

# Option 2: 4B model
language_model='google/gemma-3-4b-it'
output_dir='results_genir/caption_refinement_output_infinity-4b-flickr30k'
python genIR_CaptionImageRefinement.py \
    --input_dir='/data/flickr30k/flickr30k-images' \
    --queries_path='flickr30k_data/flickr30k_queries_2000.json' \
    --caption_model_id=$language_model \
    --comparison_model_id=$language_model \
    --output_dir=$output_dir \
    --max_images=500

# Flickr30k Eval - Text Only
# Choose one of these external_text_root options based on which model you want to evaluate
# external_text_root=results_genir/self_dialogue_caption_flickr30k-4b/captions  # 4b model
external_text_root=results_genir/self_dialogue_caption_flickr30k-12b/captions  # 12b model
# external_text_root=results_genir/visual_prediction_feedback_output_4b_flickr30k/captions  # Visual prediction feedback

python ChatIR/eval_textonly_w_incompletete.py \
    --cuda_device='0' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small_flickr30k.pth' \
    --queries_path='flickr30k_data/flickr30k_queries_2000.json' \
    --corpus_path='flickr30k_data/flickr30k_corpus_all.json' \
    --img_root='/data/flickr30k/flickr30k-images' \
    --corpus_bs=500 \
    --num_workers=64 \
    --external_text_root=$external_text_root \
    --hits_at=10

# Flickr30k Eval - Fake Images
# Choose one of these fake_images_dir options based on which model to evaluate
# fake_images_dir='results_genir/caption_refinement_output_infinity-12b-flickr30k/generated_images/'
fake_images_dir='results_genir/caption_refinement_output_infinity-4b-flickr30k/generated_images/'

python ChatIR/eval_img.py \
    --cuda_device='1' \
    --baseline='blip-zero-shot' \
    --cache_corpus='temp/corpus_blip_small_flickr30k.pth' \
    --queries_path='flickr30k_data/flickr30k_queries_2000.json' \
    --corpus_path='flickr30k_data/flickr30k_corpus_all.json' \
    --img_root='/data/flickr30k/flickr30k-images' \
    --corpus_bs=500 \
    --num_workers=64 \
    --fake_images_dir=$fake_images_dir \
    --hits_at=10




# Dev for supplymenatary
language_model='google/gemma-3-4b-it'
output_dir='results_genir/dev_supplymenatary'
python genIR_CaptionImageRefinement.py \
    --caption_model_id=$language_model \
    --comparison_model_id=$language_model \
    --output_dir=$output_dir \
    --diffusion_model='flux' \
    --max_images=1 