class ECGAugmentor:
    def __init__(self, sampling_rate=300):
        self.sampling_rate = sampling_rate

    def add_noise(self, signal, noise_factor=0.02):
        """Add Gaussian noise - reduced default noise"""
        noise = np.random.normal(0, noise_factor * np.std(signal), len(signal))
        return signal + noise

    def time_stretch(self, signal, stretch_factor=0.05):
        """Improved time stretching using interpolation"""
        factor = 1 + np.random.uniform(-stretch_factor, stretch_factor)
        original_length = len(signal)
        original_indices = np.arange(original_length)
        new_length = int(original_length / factor)
        new_indices = np.linspace(0, original_length - 1, new_length)
        interp_func = interp1d(original_indices, signal, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        stretched = interp_func(new_indices)
        if len(stretched) < original_length:
            pad_length = original_length - len(stretched)
            stretched = np.pad(stretched, (0, pad_length), mode='edge')
        elif len(stretched) > original_length:
            stretched = stretched[:original_length]

        return stretched

    def amplitude_scale(self, signal, scale_factor=0.05):
        """Reduced amplitude scaling"""
        factor = 1 + np.random.uniform(-scale_factor, scale_factor)
        return signal * factor

    def time_shift(self, signal, shift_factor=0.05):
        """Reduced time shifting"""
        max_shift = int(len(signal) * shift_factor)
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(signal, shift)

    def baseline_wander(self, signal, amplitude=0.1):
        """Add baseline wander (common ECG artifact)"""
        length = len(signal)
        freq = np.random.uniform(0.1, 0.5)
        t = np.arange(length) / self.sampling_rate
        wander = amplitude * np.sin(2 * np.pi * freq * t)
        return signal + wander

    def powerline_interference(self, signal, amplitude=0.05):
        """Add 50/60 Hz powerline interference"""
        length = len(signal)
        t = np.arange(length) / self.sampling_rate
        freq = np.random.choice([50, 60])
        interference = amplitude * np.sin(2 * np.pi * freq * t)
        return signal + interference

    def muscle_artifact(self, signal, amplitude=0.03, prob=0.3):
        """Add muscle artifact (high frequency noise in random segments)"""
        if np.random.random() > prob:
            return signal
        result = signal.copy()
        segment_length = int(len(signal) * 0.1)
        start_idx = np.random.randint(0, len(signal) - segment_length)
        muscle_noise = np.random.normal(0, amplitude, segment_length)
        result[start_idx:start_idx + segment_length] += muscle_noise

        return result

    def augment_signal(self, signal, augmentation_prob=0.5):
        """Apply random augmentations with lower probability"""
        augmented = np.array(signal, dtype=np.float32)
        augmentations = [
            (self.add_noise, 0.3),
            (self.time_stretch, 0.2),
            (self.amplitude_scale, 0.3),
            (self.time_shift, 0.2),
            (self.baseline_wander, 0.2),
            (self.powerline_interference, 0.15),
            (self.muscle_artifact, 0.15)
        ]

        for aug_func, prob in augmentations:
            if np.random.random() < prob:
                augmented = aug_func(augmented)

        return augmented

    def augment_dataset(self, X_data, y_labels, augmentation_factor=1):
        """Augment dataset with better class balance handling"""
        y_labels = np.array(y_labels)
        unique_classes, class_counts = np.unique(y_labels, return_counts=True)
        max_count = np.max(class_counts)
        augmented_X = []
        augmented_y = []
        augmented_X.extend(X_data)
        augmented_y.extend(y_labels)

        for class_label in unique_classes:
            class_indices = np.where(y_labels == class_label)[0]
            class_count = len(class_indices)

            if class_count < max_count:
                target_augmentations = int((max_count - class_count) * augmentation_factor)
            else:
                target_augmentations = int(class_count * augmentation_factor * 0.3)

            for _ in range(target_augmentations):
                idx = np.random.choice(class_indices)
                original_signal = X_data[idx]
                augmented_signal = self.augment_signal(original_signal)
                augmented_X.append(augmented_signal)
                augmented_y.append(class_label)

        return augmented_X, np.array(augmented_y)

    def pad_or_truncate(self, signals, target_length=None):
        """Pad or truncate signals to same length"""
        if target_length is None:
            lengths = [len(signal) for signal in signals]
            target_length = int(np.median(lengths))

        processed_signals = []
        for signal in signals:
            if len(signal) < target_length:
                padded = np.pad(signal, (0, target_length - len(signal)), mode='constant')
                processed_signals.append(padded)
            elif len(signal) > target_length:
                processed_signals.append(signal[:target_length])
            else:
                processed_signals.append(signal)

        return np.array(processed_signals)

    def validate_augmentation(self, original_signal, augmented_signal, max_change=0.3):
        """Validate that augmentation doesn't change signal too much"""
        correlation = np.corrcoef(original_signal, augmented_signal)[0, 1]

        orig_amplitude = np.max(original_signal) - np.min(original_signal)
        aug_amplitude = np.max(augmented_signal) - np.min(augmented_signal)
        amplitude_change = abs(aug_amplitude - orig_amplitude) / orig_amplitude

        return correlation > 0.7 and amplitude_change < max_change


# TASK 3: DATA AUGMENTATION (ImprovedSTFTModel)
print("\n" + "TASK 3: DATA AUGMENTATION (ImprovedSTFT)")

augmentor = ECGAugmentor()
X_train_aug, y_train_aug = augmentor.augment_dataset(X_train_split, y_train_split, augmentation_factor=0.5)
print(f"Augmented dataset size: {len(X_train_aug)} (original: {len(X_train_split)})")
train_dataset_aug = ECGDataset(X_train_aug, y_train_aug, augmentor=augmentor, is_training=True)
train_loader_aug = DataLoader(train_dataset_aug, batch_size=32, shuffle=True, collate_fn=collate_fn)
print("Training STFT...")
model_aug = ImprovedSTFTModel(num_classes=4)
trainer_aug = ECGTrainer(model_aug, device=device, name_prefix='augment_')
print(f'Model initialized with {sum(p.numel() for p in model_aug.parameters())} parameters')
print("Training with augmented data...")
history_aug = trainer_aug.train(train_loader_aug, val_loader, epochs=100, patience=50)

test_dataset = ECGDataset(X_test, np.zeros(len(X_test)), is_training=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
generate_base_predictions(model_aug, test_loader, device, pre='augment_')

plot_training_history(history_aug, model_name="AugmentedSTFTModel", save_path="augmented_training_history.png")

