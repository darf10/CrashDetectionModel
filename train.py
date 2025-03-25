import os
import argparse
from pipeline.training_middleware import PipelineTrainingMiddleware
from model.Model_Config import ModelConfig


def parse_arguments():
    """
    Parse command-line arguments for training configuration
    """
    parser = argparse.ArgumentParser(description='Train Accident Detection LSTM Model')

    # Video data arguments
    parser.add_argument('--normal_videos', nargs='+',
                        help='Paths to normal driving condition videos')
    parser.add_argument('--accident_videos', nargs='+',
                        help='Paths to accident videos')

    # Model configuration arguments
    parser.add_argument('--input_size', type=int, default=10,
                        help='Number of features per time step')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden layer size for LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')

    # Model saving arguments
    parser.add_argument('--model_path', default='models/accident_classifier.pth',
                        help='Path to save trained model')
    parser.add_argument('--scaler_path', default='models/feature_scaler.joblib',
                        help='Path to save feature scaler')

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set up model configuration
    config = ModelConfig(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_path=args.model_path,
        scaler_path=args.scaler_path
    )

    # Create training middleware
    training_middleware = PipelineTrainingMiddleware(config)

    # Prepare video paths and labels
    # Assumes first half of videos are normal, second half are accidents
    normal_videos = args.normal_videos or []
    accident_videos = args.accident_videos or []

    if not (normal_videos and accident_videos):
        print("Error: Please provide both normal and accident video paths")
        return

    # Combine videos with corresponding labels
    video_paths = normal_videos + accident_videos
    labels = [0] * len(normal_videos) + [1] * len(accident_videos)

    # Train the model
    print("Starting model training...")
    training_history = training_middleware.train_from_videos(video_paths, labels)

    if training_history:
        print("Training completed successfully!")
        print("Training Loss:", training_history['train_loss'][-1])
        print("Training Accuracy:", training_history['train_accuracy'][-1])
    else:
        print("Training failed or no data collected.")


if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs('model', exist_ok=True)

    main()
