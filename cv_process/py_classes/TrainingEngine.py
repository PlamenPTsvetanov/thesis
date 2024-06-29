from RecognitionSetup import Setup

from Training import Training

class TrainingEngine:
    @staticmethod
    def run():
        meta = Setup()
        meta.setup()

    @staticmethod
    def train():
        TrainingEngine.run()
        train = Training()
        train.train_model()


if __name__ == "__main__":
    TrainingEngine.train()



