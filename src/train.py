import numpy as np
from preprocessing import preprocess
from feature_extraction import extract_features, segment_signal
from model import build_model
from sklearn.preprocessing import StandardScaler

# Dummy loader (replace with NinaPro loader)
def load_data():
    X = np.random.randn(10000, 12)  # EMG channels
    y_gesture = np.random.randint(0, 40, 10000)
    y_force = np.random.rand(10000)
    return X, y_gesture, y_force

def prepare_features(X):
    segments = segment_signal(X)
    features = [extract_features(w) for w in segments]
    return np.array(features)

def main():
    X, y_gesture, y_force = load_data()

    X = preprocess(X)
    X_feat = prepare_features(X)

    y_gesture = y_gesture[:len(X_feat)]
    y_force = y_force[:len(X_feat)]

    scaler = StandardScaler()
    X_feat = scaler.fit_transform(X_feat)

    model = build_model(input_dim=X_feat.shape[1], num_classes=40)

    model.fit(
        X_feat,
        {'gesture': y_gesture, 'force': y_force},
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    model.save("emg_model.h5")

if __name__ == "__main__":
    main()
