from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/media/quad/0A9AD66165F33762/ImageCaption/Datasets/MSCOCO/dataset_coco.json',
                       image_folder='/media/quad/0A9AD66165F33762/ImageCaption/Datasets/MSCOCO/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/media/quad/0A9AD66165F33762/ImageCaption/Datasets/MSCOCO/',
                       max_len=50)
