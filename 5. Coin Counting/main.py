import argparse
from hse_cv.util.resize_images import resize_images
from hse_cv.coin_counting.train_classifier import train_model
from hse_cv.coin_counting.predict import predict_sum

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Global arguments
    parser.add_argument('-c', help='Command to execute')
    # Resize images arguments
    parser.add_argument('-input_dir', help='Path to images folder')
    parser.add_argument('-scaling_factor', help='Scaling factor')
    parser.add_argument('-output_dir', help='Output directory')
    # Predict command arguments
    parser.add_argument('-image', help='Path to image')
    args = parser.parse_args()
    command = args.c
    if command == 'resize':
        resize_images(args.input_dir, float(args.scaling_factor), args.output_dir)
    elif command == 'train':
        train_model(args.input_dir)
    elif command == 'predict':
        predict_sum(args.image)
